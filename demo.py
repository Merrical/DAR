import torch
from EfficientNet_2d.EfficientNet_2d import get_pretrained_EfficientNet, get_pretrained_DAR


if __name__ == "__main__":
    # Phase 1:
    # pre-train prd-net, cf-net, and lr-net on CR-set, IC-set, and LR-set, respectively, and save the pre-trained model
    prd_net = get_pretrained_EfficientNet(num_classes=5)
    cf_net = get_pretrained_EfficientNet(num_classes=5)
    lr_net = get_pretrained_EfficientNet(num_classes=5)

    # Phase 2:
    # fine-tune dar on CR-set
    prd_params_path = "../your_checkpoint_path/prd_net_checkpoint.pth"
    cf_params_path = "../your_checkpoint_path/cf_net_checkpoint.pth"
    lr_params_path = "../your_checkpoint_path/lr_net_checkpoint.pth"

    prd_params = torch.load(prd_params_path)
    cf_params = torch.load(cf_params_path)
    lr_params = torch.load(lr_params_path)

    model = get_pretrained_DAR(prd_params, cf_params, lr_params, num_classes=5)

    # prediction
    imgs = torch.rand([4, 3, 224, 224])
    prd_preds, cf_preds, lr_preds = model(imgs)
    _, preds = torch.softmax(prd_preds, dim=1).max(dim=1)
