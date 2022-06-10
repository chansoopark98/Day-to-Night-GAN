from utils.datasets import Dataset
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import time

parser = argparse.ArgumentParser()
parser.add_argument("--model_prefix",     type=str,   help="Model name", default='New_code')
parser.add_argument("--batch_size",     type=int,   help="배치 사이즈값 설정", default=8)
parser.add_argument("--epoch",          type=int,   help="에폭 설정", default=50)
parser.add_argument("--lr",             type=float, help="Learning rate 설정", default=0.001)
parser.add_argument("--optimizer",     type=str,   help="Optimizer", default='adam')
parser.add_argument("--model_name",     type=str,   help="저장될 모델 이름",
                    default=str(time.strftime('%m%d', time.localtime(time.time()))))
parser.add_argument("--dataset_dir",    type=str,   help="데이터셋 다운로드 디렉토리 설정", default='./datasets/')
parser.add_argument("--result_dir",    type=str,   help="Validation 결과 이미지 저장 경로", default='./result_dir/')
parser.add_argument("--checkpoint_dir", type=str,   help="모델 저장 디렉토리 설정", default='./checkpoints/')
parser.add_argument("--tensorboard_dir",  type=str,   help="텐서보드 저장 경로", default='tensorboard/')
parser.add_argument("--use_weightDecay",  type=bool,  help="weightDecay 사용 유무", default=False)
parser.add_argument("--save_weight",  type=bool, help="학습 가중치 저장", default=True)
parser.add_argument("--save_frac",  type=int, help="학습 가중치 저장", default=3)
parser.add_argument("--load_weight",  help="가중치 로드", action='store_true')
parser.add_argument("--mixed_precision",  type=bool,  help="mixed_precision 사용", default=False)

args = parser.parse_args()


if __name__ == '__main__':
    print('main')

    config = {
        'image_size': (512, 512),
        'gen_input_channel': 1,
        'gen_output_channel': 2,
        'dis_input_channel': 3
    }

    dataset_config = Dataset(
        data_dir=args.dataset_dir,
        image_size=config['image_size'],
        batch_size=args.batch_size,
    )

    train_data = dataset_config.get_testData(dataset_config.train_data)
    steps_per_epoch = dataset_config.number_train // args.batch_size

    pbar = tqdm(train_data, total=steps_per_epoch, desc='Batch', leave=True, disable=False)

    for rgb, label in pbar:
        
        for batch in range(rgb.shape[0]):
            print(label[batch])
            plt.imshow(rgb[batch])
            plt.show()