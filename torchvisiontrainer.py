import torch
import torchvision
import os
from torch.utils.data import DataLoader
from torchvision.models.detection import ssdlite320_mobilenet_v3_large # chnge this based on your model
from torchvision.transforms import functional as F
from PIL import Image

class CustomCocoDataset(torch.utils.data.Dataset):
    def __init__(self, root, annFile, transforms=None):
        from pycocotools.coco import COCO
        self.root = root
        self.coco = COCO(annFile)
        self.ids = list(self.coco.imgs.keys())
        self.transforms = transforms

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)

        path = coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.root, path)).convert("RGB")

        boxes, labels, areas, iscrowd = [], [], [], []
        for ann in anns:
            xmin, ymin, w, h = ann['bbox']
            boxes.append([xmin, ymin, xmin + w, ymin + h])
            labels.append(ann['category_id'])
            areas.append(ann['area'])
            iscrowd.append(ann.get('iscrowd', 0))

        # Always create a target, even if empty
        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32),
            "labels": torch.as_tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64),
            "area": torch.as_tensor(areas, dtype=torch.float32) if areas else torch.zeros((0,), dtype=torch.float32),
            "iscrowd": torch.as_tensor(iscrowd, dtype=torch.int64) if iscrowd else torch.zeros((0,), dtype=torch.int64),
            "image_id": torch.tensor([img_id])
        }

        if self.transforms is not None:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.ids)

def get_transform():
    return F.to_tensor


def collate_fn(batch):
    return tuple(zip(*batch))

def train_model(train_root, train_ann, classes, epochs=10, batch_size=2, lr=0.001):
    dataset = CustomCocoDataset(train_root, train_ann, transforms=F.to_tensor)
    print(len(dataset))
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    model = ssdlite320_mobilenet_v3_large(weights=None, num_classes=classes) #change this based on ur model too.

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=0.0005)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for batch_idx, (imgs, targets) in enumerate(data_loader):
            imgs = list(img.to(device) for img in imgs)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(imgs, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            batch_loss = losses.item()
            epoch_loss += batch_loss

            print(f"Epoch [{epoch+1}/{epochs}] Batch [{batch_idx+1}/{len(data_loader)}] Loss: {batch_loss:.4f}")

    print("Training complete.")
    return model

if __name__ == "__main__":
    # CHANGE THESE PATHS TO MATCH YOUR COCO EXPORT
    train_root = "roboflow/train/"
    train_ann = "roboflow/train/_annotations.coco.json"
    num_classes = 2  #(make sure to update this!)

    model = train_model(train_root, train_ann, num_classes, epochs=150, batch_size=16)

    torch.save(model.state_dict(), "model.pth")
    print("Model saved to model.pth")

