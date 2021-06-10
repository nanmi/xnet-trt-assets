import os
import sys
import torch
import struct
import argparse

sys.path.insert(0, os.getcwd())

def GenerateWTSwithPT(input_pt_file, output_wts_file):
    # Initialize device
    device = "cpu"

    # Load model
    if input_pt_file.endswith(".pt"):
        assert('model' in ['epoch', 'best_fitness', 'training_results', 'model', 'optimizer'])
        model = torch.load(input_pt_file, map_location=device)['model'].float()  # load to FP32
        model.to(device).eval()
        print('-------Converting-------')
        print('[INFO]   Generating ... ', output_wts_file)

        with open(output_wts_file, 'w') as f:
            f.write('{}\n'.format(len(model.state_dict().keys())))
            for k, v in model.state_dict().items():
                vr = v.reshape(-1).cpu().numpy()
                f.write('{} {} '.format(k, len(vr)))
                for vv in vr:
                    f.write(' ')
                    f.write(struct.pack('>f',float(vv)).hex())
                f.write('\n')
        print('-------Converted-------')

    # ===========================================================================

    elif input_pt_file.endswith(".pth"):
        assert('state_dict' in ['epoch', 'state_dict'])
        model = torch.load(input_pt_file, map_location=device)#['state_dict']
        print(model)
        print('-------Converting-------')
        print('[INFO]   Generating ... ', output_wts_file)

        with open(output_wts_file, 'w') as f:
            f.write('{}\n'.format(len(model.items())))
            for key, value in model.items():
                vr = value.reshape(-1).cpu().numpy()
                f.write('{} {} '.format(key, len(vr)))
                for vv in vr:
                    f.write(' ')
                    f.write(struct.pack('>f', float(vv)).hex())
                f.write('\n')
        print('-------Converted-------')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pt', type=str, default='', help='initial .pt/.pth weights file path')
    parser.add_argument('--wts', type=str, default='', help='output .wts weights file path')
    opt = parser.parse_args()
    GenerateWTSwithPT(opt.pt, opt.wts)