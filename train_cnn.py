import os
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from transformers import ViTForImageClassification, TrainingArguments, Trainer

class ASLDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.images = []
        self.labels = []
        
        for cls_name in self.classes:
            cls_dir = os.path.join(root_dir, cls_name)
            for img_name in os.listdir(cls_dir):
                self.images.append(os.path.join(cls_dir, img_name))
                self.labels.append(self.class_to_idx[cls_name])
        
    def __len__(self):
        return len(self.images)
        
    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return {"pixel_values": image, "labels": label}

def prepare_data(data_dir, batch_size=8):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    dataset = ASLDataset(root_dir=data_dir, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def train_model(train_loader, val_loader):
    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224",
        num_labels=36,  # Updated to 36 classes (A-Z and 0-9)
        ignore_mismatched_sizes=True
    )
    
    training_args = TrainingArguments(
        output_dir="./hgr-vit-finetuned",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=2,
        learning_rate=3e-5,
        num_train_epochs=10,
        fp16=True,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir="./logs",
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
    )
    
    def compute_metrics(eval_pred):
        predictions = eval_pred.predictions.argmax(-1)
        labels = eval_pred.label_ids
        return {"accuracy": (predictions == labels).mean()}
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_loader.dataset,
        eval_dataset=val_loader.dataset,
        compute_metrics=compute_metrics,
    )
    
    trainer.train()
    
    model.save_pretrained("./hgr-vit-finetuned")
    print("Fine-tuned model saved at ./hgr-vit-finetuned")
    
    return model

def quantize_model(model, val_loader):
    model.eval()
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    model_prepared = torch.quantization.prepare(model)
    
    print("Calibrating quantized model...")
    with torch.no_grad():
        for batch in val_loader:
            pixel_values = batch['pixel_values']
            _ = model_prepared(pixel_values)
    
    model_quantized = torch.quantization.convert(model_prepared)
    
    torch.save(model_quantized.state_dict(), "hgr_vit_quantized.pth")
    print("Quantized model saved at hgr_vit_quantized.pth")
    
    return model_quantized

def main(data_dir):
    train_loader = prepare_data(os.path.join(data_dir, "train_augmented"))
    val_loader = prepare_data(os.path.join(data_dir, "val"))
    
    fine_tuned_model = train_model(train_loader=train_loader, val_loader=val_loader)
    optimized_model = quantize_model(fine_tuned_model, val_loader)

if __name__ == "__main__":
    data_dir = "dataset"  # This should be the directory containing train, val, and test subdirectories
    main(data_dir)
