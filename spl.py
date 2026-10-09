"""SPL objective; copied algebra and update order from the repaired CRPL B0."""
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def soft_cross_entropy_loss(predictions, soft_targets):
    return -torch.sum(soft_targets * F.log_softmax(predictions, dim=1), dim=1).mean()


def calc_distance(a, b):
    # Upstream squared L2 on unnormalized final image embeddings, without clamp.
    return torch.sum(a**2, dim=1, keepdim=True) + torch.sum(b**2, dim=1) - 2 * torch.matmul(a, b.t())


class LossValley:
    def __init__(self, n_converge=20):
        from collections import deque
        self.n_converge = n_converge
        self.converge_Q = deque(maxlen=n_converge)
        self.is_converged = False
        self.converged_step = None

    def update(self, loss, step):
        if self.is_converged:
            return
        self.converge_Q.append(loss)
        if len(self.converge_Q) == self.n_converge and np.argmin(self.converge_Q) == 0:
            self.is_converged = True
            self.converged_step = step

    def state_dict(self):
        return {"queue": list(self.converge_Q), "is_converged": self.is_converged,
                "converged_step": self.converged_step}

    def load_state_dict(self, state):
        self.converge_Q.clear()
        self.converge_Q.extend(state["queue"])
        self.is_converged = state["is_converged"]
        self.converged_step = state["converged_step"]


def spl_step(prompt, encoder, clip_model, source_data, source_label, source_domain,
             target_data, base_tokens, running_means, running_count, valley, step, args):
    """Target input has no labels; source CE and teacher soft CE remain unchanged."""
    pooled_prompts, target_prompts = prompt()
    scale = encoder.logit_scale.exp()
    source_img, source_text, source_raw = encoder(source_data, pooled_prompts, prompt.tokenized_prompts)
    pooled_logits = source_img @ source_text.t()
    pooled_loss = F.cross_entropy(scale * pooled_logits, source_label)
    valley.update(pooled_loss.item(), step)
    source_texts, features, labels = [], [], []
    n_domains = running_means.shape[0]
    for i in range(n_domains):
        text_features = encoder.forward_txt(prompt.forward_source(i), prompt.tokenized_prompts)
        source_texts.append(text_features)
        mask = source_domain == i
        features.append(source_img[mask])
        labels.append(source_label[mask])
        raw = source_raw[mask]
        for cls in torch.unique(labels[i]):
            cls_features = raw[labels[i] == cls]
            running_means[i][cls] = running_means[i][cls] * running_count[i][cls] + cls_features.sum(0)
            running_count[i][cls] += cls_features.shape[0]
            running_means[i][cls] /= running_count[i][cls]
    source_losses = [F.cross_entropy(scale * (f @ t.t()), y)
                     for f, t, y in zip(features, source_texts, labels)]
    source_avg = sum(source_losses) / n_domains
    target_img, target_text, target_raw = encoder(target_data, target_prompts, prompt.tokenized_prompts)
    target_logits = target_img @ target_text.t()
    _, base_text = clip_model(target_data, base_tokens)
    teacher_logits = target_img @ base_text.t()
    teacher_logits += target_img @ source_text.t()
    n_pseudo = 2
    weighted_enabled = (running_count > 0).sum().sum() == running_count.numel()
    weights = None
    if weighted_enabled:
        distance = torch.zeros(n_domains, target_img.shape[0], running_means.shape[1], device=args.device)
        for i in range(n_domains):
            distance[i] = calc_distance(target_raw, running_means[i])
        weights = nn.Softmax(dim=0)(-distance * args.w_scale).detach()
        for i in range(n_domains):
            teacher_logits += weights[i] * (target_img @ source_texts[i].t())
        n_pseudo += 1
    teacher_logits /= n_pseudo
    teacher = (scale * teacher_logits).softmax(-1).detach()
    target_loss = soft_cross_entropy_loss(scale * target_logits, teacher)
    # Keep the two original source additions and their original operation order.
    total_loss = pooled_loss + source_avg
    total_loss += args.t_weight * target_loss
    with torch.no_grad():
        student_log = F.log_softmax(scale * target_logits, dim=-1)
        teacher_log = teacher.clamp_min(torch.finfo(teacher.dtype).tiny).log()
        metrics = {"train/loss_total": total_loss.item(), "train/loss_source_pooled": pooled_loss.item(),
                   "train/loss_source_domain_avg": source_avg.item(), "train/loss_target_soft": target_loss.item(),
                   "train/teacher_entropy": -(teacher * teacher_log).sum(-1).mean().item(),
                   "train/teacher_confidence_max_mean": teacher.max(-1).values.mean().item(),
                   "train/student_teacher_agreement": (target_logits.argmax(-1) == teacher.argmax(-1)).float().mean().item(),
                   "train/student_teacher_kl": (teacher * (teacher_log - student_log)).sum(-1).mean().item(),
                   "train/source_centroid_coverage": (running_count > 0).float().mean().item(),
                   "train/weighted_teacher_enabled": int(weighted_enabled)}
        metrics.update({f"train/loss_source_domain_{i}": loss.item() for i, loss in enumerate(source_losses)})
        if weights is not None:
            metrics["routing/source_weight_entropy"] = -(weights * weights.clamp_min(1e-38).log()).sum(0).mean().item()
            metrics["routing/source_weight_max_mean"] = weights.max(0).values.mean().item()
    return total_loss, target_text, metrics


@torch.no_grad()
def evaluate(loader, encoder, text_features, batch_size, device):
    correct = total = 0
    for images, labels in loader:
        logits = encoder.forward_img(images.to(device)) @ text_features.t()
        correct += (logits.argmax(1) == labels.to(device)).sum().item()
        # All batches are full because drop_last=True, matching B0.
        if len(images) != batch_size:
            raise ValueError("Evaluation loader must follow the baseline dropped-tail protocol")
        total += batch_size
    return correct / total
