import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
from models import VisionTransformer, VITC  # Import the model without running it

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vision_transformer = VisionTransformer(
        image_size=(32, 32),
        patch_size=4,
        num_classes=10,
        num_heads=8,
        mlp_ratio=0.8,
        norm_layer=nn.LayerNorm,
        embed_norm_layer=nn.LayerNorm,
        final_norm_layer=nn.LayerNorm
    )

    vitc = VITC(vit_model=vision_transformer)

    checkpoint = torch.load('vitc_checkpoint.pth', map_location=device)
    vitc.load_state_dict(checkpoint['model_state_dict'])
    vitc = vitc.to(device)
    vitc.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    batch_size = 4
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    with torch.no_grad():
        correct = 0
        total = 0
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = vitc(inputs)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(f'Accuracy on the test set: {100 * correct / total:.2f}%')

if __name__ == '__main__':
    main()
