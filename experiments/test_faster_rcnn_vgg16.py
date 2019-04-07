# model_loader.py to load my-faster-rcnn model

import settings
import torch
import torchvision

def loadmodel(hook_fn):
    if settings.MODEL_FILE is None:
        model = torchvision.models.__dict__[settings.MODEL](pretrained=True)
    else:
        checkpoint = torch.load(settings.MODEL_FILE)
        checkpoint = dict(checkpoint['model'])
        if type(checkpoint).__name__ == 'OrderedDict' or type(checkpoint).__name__ == 'dict':
            model = torchvision.models.__dict__[settings.MODEL](num_classes=settings.NUM_CLASSES)
            state_dict = checkpoint
            feature_name = settings.FEATURE_NAMES[-1]
            for key in list(state_dict):
                if key.startswith('CNN.'):
                    state_dict[key.lstrip('CNN.')] = state_dict.pop(key)
                else:
                    state_dict.pop(key)
            getattr(model, feature_name).load_state_dict(state_dict)
        else:
            model = checkpoint
    for name in settings.FEATURE_NAMES:
        model._modules.get(name).register_forward_hook(hook_fn)
    if settings.GPU:
        model.cuda()
    model.eval()
    return model
