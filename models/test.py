import torchvision
encoder=getattr(torchvision.models, 'resnet50')(pretrained=False)
print(encoder.inplanes)