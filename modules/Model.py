import torch
import torch.optim as optim

from monai.networks.nets import UNet
from monai.losses import DiceLoss
from monai.metrics import DiceMetric


class Model:

    def __init__(self):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = UNet(
            spatial_dims=3,
            in_channels=4,
            out_channels=3,
            channels=(16,32,64,128,256),
            strides=(2,2,2,2),
            num_res_units=2
        ).to(self.device)

        self.loss_function = DiceLoss(to_onehot_y=True, softmax=True)

        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)


    def Train(self, Dataset):

        for epoch in range(50):

            self.model.train()
            epoch_loss = 0

            for batch_data in Dataset:

                inputs = batch_data["image"].to(self.device)
                labels = batch_data["label"].to(self.device)

                self.optimizer.zero_grad()

                outputs = self.model(inputs)

                loss = self.loss_function(outputs, labels)

                loss.backward()

                self.optimizer.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(Dataset)

            print(f"Epoch {epoch} | Loss: {avg_loss:.4f}")

        torch.save(self.model.state_dict(), "model.pth")


    def Test(self, Dataset):

        self.model.load_state_dict(torch.load("model.pth", map_location=self.device))
        self.model.eval()

        dice_metric = DiceMetric(include_background=False, reduction="mean")

        total_correct = 0
        total_voxels = 0

        with torch.no_grad():

            for batch_data in Dataset:

                images = batch_data["image"].to(self.device)
                labels = batch_data["label"].to(self.device)

                outputs = self.model(images)

                preds = torch.argmax(outputs, dim=1, keepdim=True)

                dice_metric(preds, labels)

                correct = (preds == labels).sum().item()
                total = torch.numel(labels)

                total_correct += correct
                total_voxels += total

        dice_score = dice_metric.aggregate().item()
        accuracy = total_correct / total_voxels

        print("Test Dice Score:", dice_score)
        print("Voxel Accuracy:", accuracy)

        dice_metric.reset()


    def Use(self, patient_tensor):

        self.model.load_state_dict(torch.load("model.pth", map_location=self.device))
        self.model.eval()

        with torch.no_grad():

            test_input = patient_tensor.unsqueeze(0).to(self.device)

            prediction = self.model(test_input)

            prediction = torch.argmax(prediction, dim=1)

        return prediction