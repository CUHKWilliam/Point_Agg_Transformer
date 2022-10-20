import spconv.pytorch as spconv
import torch

model_path = "./best_fs_geoformer_scannet_fold0.pth"
out_path = "./best_fs_geoformer_scannet_fold0_converted.pth"
if __name__ == "__main__":
    ckpt = torch.load(model_path)
    state_dict = torch.load(model_path)["state_dict"]

    for key in state_dict.keys():
        if ('unet' in key or "input_conv" in key) and 'weight' in key and state_dict[key].dim() == 5:
            print(key)
            # print(pretrained_model['net'][key].shape)
            state_dict[key] = state_dict[key].permute([4, 0, 1, 2, 3])

    ckpt["state_dict"] = state_dict
    torch.save(ckpt, out_path)
    # for key in pretrained_model['optim']['state'].keys():
    #     if pretrained_model['optim']['state'][key]['exp_avg'].dim() == 5:
    #         pretrained_model['optim']['state'][key]['exp_avg'] = pretrained_model['optim']['state'][key][
    #             'exp_avg'].permute([4, 0, 1, 2, 3])
    #         pretrained_model['optim']['state'][key]['exp_avg_sq'] = pretrained_model['optim']['state'][key]['exp_avg_sq'].permute([4, 0, 1, 2, 3])
