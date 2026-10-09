"""SPL loaders: retain upstream sampling/coverage without target-GT diagnostics."""
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets

from clip_custom import clip
from style import UnlabelledImages


class Pseudolabeldata(Dataset):
    def __init__(self, path, clip_model, transform, args, classnames):
        dataset = UnlabelledImages(path, transform)
        # Same order and batch size as upstream ImageFolder scan. No labels.
        loader = DataLoader(dataset, num_workers=args.num_workers, batch_size=args.batch_size,
                            shuffle=False, pin_memory=args.pin_memory, drop_last=False)
        text = clip.tokenize([f"A photo of a {name}" for name in classnames]).to(args.device)
        scale = clip_model.logit_scale.exp()
        self.instances = []
        with torch.no_grad():
            for images in loader:
                images = images.to(args.device)
                image_features, text_features = clip_model(images, text)
                probs = (scale * image_features @ text_features.t()).softmax(-1)
                confidence, pseudo_labels = probs.max(-1)
                for image, label, conf in zip(images.cpu(), pseudo_labels.cpu(), confidence.cpu()):
                    self.instances.append((image, int(label) if conf > args.threshold else -1))

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, index):
        return self.instances[index]


def load_pseudo_label_data(path, preprocess, clip_model, args, classnames):
    return DataLoader(Pseudolabeldata(path, clip_model, preprocess, args, classnames),
                      num_workers=args.num_workers, batch_size=args.batch_size,
                      shuffle=True, pin_memory=args.pin_memory, drop_last=True)


def load_data(path, preprocess, args):
    """Evaluation only: preserve author's shuffled, dropped-tail protocol."""
    return DataLoader(datasets.ImageFolder(path, transform=preprocess), num_workers=args.num_workers,
                      batch_size=args.batch_size, shuffle=True, pin_memory=args.pin_memory, drop_last=True)
