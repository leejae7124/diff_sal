import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from datasets.meta_data import MetaDataset
from util.registry import DATASET_REGISTRY


DATASET_REGISTRY.register_module()
class UCFDataset(MetaDataset):

    def __init__(self,
                 path_data,
                 len_snippet,
                 mode="train",
                 img_size=(224, 384),
                 gt_length=1,
                 alternate=1
                 ):
        super(UCFDataset, self).__init__(path_data, len_snippet, mode, img_size, alternate, gt_length)

        self.vid_list = self.get_video_list(mode) #memor 재작성 필요. mode 필요 없음

        if self.mode == "train":
            self.path_data = os.path.join(self.path_data, "training")
            self.list_num_frame = []
            for v in self.vid_list:
                tmp_video_len = len(os.listdir(os.path.join(self.path_data, v, "images")))
                for i in range(0, tmp_video_len - self.alternate * self.len_snippet, self.skip_window):
                    self.list_num_frame.append((v, i))
        else:
            self.path_data = os.path.join(self.path_data, "testing") #경로 설정
            self.list_num_frame = []
            self.list_video_frames = {}
            for v in self.vid_list: #비디오 이름 리스트가 들어있음. 비디오 이름별로 for문 진행
                tmp_video_len = len(os.listdir(os.path.join(self.path_data, v, "images"))) # ~/testing/비디오 이름/images/프레임들. "비디오 길이"
                if tmp_video_len < self.alternate * self.len_snippet:
                    continue
                for i in range(0, tmp_video_len - self.alternate * self.len_snippet, gt_length): #비디오 길이 초과하지 않을 때까지만.
                    self.list_num_frame.append((v, i))
                self.list_num_frame.append((v, tmp_video_len - self.len_snippet))
                self.list_video_frames[v] = tmp_video_len

        print(f"UCF {mode} dataset loaded! {len(self.list_num_frame)} data items!")
        #정리: 초기화 할 때, 프레임을 로드한다.

    def __getitem__(self, idx):
        (file_name, start_idx) = self.list_num_frame[idx] #각 비디오의 스니펫을 시작하는 프레임 인덱스와 비디오 이름이 튜플로 담겨있음. ("video1", 0), ("video1", 10), ...

        path_clip = os.path.join(self.path_data, file_name, "images") #VideoSalPrediction/ucf/testing/video1/images
        path_annt = os.path.join(self.path_data, file_name, "maps") #VideoSalPrediction/ucf/testing/video1/maps

        data = {'rgb': []}
        target = {'salmap': []}

        # TODO 继续测试在16帧参数的数据集上的效果
        if self.len_snippet > 16: #16개의 프레임만 선택한다.
            frame_lens = 16
            indices = [start_idx + self.alternate * i + 1 for i in range(frame_lens)] #하나씩 건너뛰면서 선택함. 시간적 길이는 동일하지만 sampling rate를 다르게 처리!
        else:
            indices = [start_idx + self.alternate * i + 1 for i in range(self.len_snippet) ]
            
        clip_img = [] #프레임을 담는다.
        vid_name, vid_index =file_name.split("-")[:-1], file_name.split("-")[-1]
        vid_name = file_name.strip(f"-{vid_index}")
        median = int(np.mean(np.array(indices)))
        for i in indices:
            img = Image.open(os.path.join(path_clip, '{}_{}_{:03d}.png'.format(vid_name, vid_index, i))).convert('RGB') #프레임 파일(이름) 처리
            clip_img.append(self.img_transform(img))
        clip = torch.stack(clip_img, 0).permute(1, 0, 2, 3) #데이터의 맨 앞에 차원을 추가. N: 스니펫의 길이를 추가로 한다. -> 모델의 입력에 맞게 조정하는 것.
        clip_img = torch.FloatTensor(clip) #float 형태로 변환

        def get_center_slice(arr, length):
            center = len(arr) // 2  # 获取数组的中心位置索引
            start = center - length // 2  # 计算起始索引
            end = start + length  # 计算结束索引
            return arr[start:end]

        # 加载gt maps需要特别关注它的数据分布
        clip_gt = None
        if self.mode != "save" and self.mode != "test":
            gt_sequence_list = get_center_slice(indices, self.gt_length)
            gt_maps_list = []
            for gt_index in gt_sequence_list:
                mid_img_name = '{}_{}_{:03d}.png'.format(vid_name, vid_index, gt_index)
                gt_index_path = os.path.join(path_annt, mid_img_name)
                gt_maps_list.append(self.load_gt_PIL(gt_index_path))
            gt_maps = torch.stack(gt_maps_list, 0).permute(1, 0, 2, 3).squeeze(1)
            clip_gt = gt_maps

        data['rgb'] = clip_img
        data["video_id"] = file_name
        data["video_index"] = file_name     # 用于预测
        data["gt_index"] = torch.tensor(gt_sequence_list)
        target['salmap'] = clip_gt


        return data, target


if __name__ == '__main__':
    
    train_data = UCFDataset(path_data="VideoSalPrediction/ucf",
                     len_snippet=32, 
                     mode="test")

    tmp_data = train_data.__getitem__(0)
    print(train_data.__len__())

    from torch.utils.data import DataLoader

    #한 배치씩 데이터를 업로드한다.
    data_loader = DataLoader(train_data, batch_size=16, num_workers=0) #batch_size=16은, 하나의 배치에 16개의 snippet이 포함된다는 뜻
    #DataLoader의 처리 흐름
    # 1. Dataset의 __getitem__ 호출
    # 2. 배치 생성
    #   - batch_size=16이라면, __getitem__ 메서드를 16번 호출하여, 이를 하나의 배치로 만든다.
    for batch, target in data_loader:
        # print(batch["rgb"].shape)
        # print(target["salmap"].shape)
        print(batch["video_id"], batch["gt_index"])