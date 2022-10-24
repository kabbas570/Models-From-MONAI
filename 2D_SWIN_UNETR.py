from monai.networks.nets import SwinUNETR
import  torch
model = SwinUNETR(
        img_size=(224,224),
        in_channels=1,
        out_channels=1,
        feature_size=48,
        use_checkpoint=False,spatial_dims=2
    )

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

from torchsummary import summary
#model = model()
model.to(device=DEVICE,dtype=torch.float)
summary(model, (1, 224, 224))
