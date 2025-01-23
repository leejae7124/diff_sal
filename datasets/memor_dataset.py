import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from datasets.meta_data import MetaDataset
from util.registry import DATASET_REGISTRY

DATASET_REGISTRY.register_module()
class MEmoRDataset(MetaDataset):

    def __init__(self,
                 path_data,
                 len_snippet,
                 mode="train",
                 img_size=(224, 384),
                 gt_length=1,
                 alternate=1
                 ):
        super(MEmoRDataset, self).__init__(path_data, len_snippet, mode, img_size, alternate, gt_length)

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

        print(f"MEmoR {mode} dataset loaded! {len(self.list_num_frame)} data items!")
        #정리: 초기화 할 때, 프레임을 로드한다.

    def __getitem__(self, idx):

        (file_name, start_idx) = self.list_num_frame[idx] #각 비디오의 스니펫(len_snippet = 32)을 시작하는 프레임 인덱스와 비디오 이름이 튜플로 담겨있음. ("video1", 0), ("video1", 32), ...
        # for idx, (file_name, start_idx) in enumerate(self.list_num_frame):
        #     print(f"Index {idx}: File {file_name}, Start Index {start_idx}")

        # print("file name!!!!! ", file_name)
        path_clip = os.path.join(self.path_data, file_name, "images") #VideoSalPrediction/ucf/testing/video1/images
        path_annt = os.path.join(self.path_data, file_name, "maps") #VideoSalPrediction/ucf/testing/video1/maps
        # print(file_name, "  FFFFFFFFFFFFFFFilename")

        data = {'rgb': []} #모델의 입력 데이터(dictionary)
        target = {'salmap': []}

        # TODO 继续测试在16帧参数的数据集上的效果

        if self.len_snippet > 16: #16개의 프레임만 선택한다. 1. alternate = 1: 처음부터 16개 프레임 2. alternate = 2: 1개씩 뛰어넘으며 16개의 프레임
            frame_lens = 16
            indices = [start_idx + self.alternate * i for i in range(frame_lens)]
            # print("indices: ", indices, file_name)
        else:
            indices = [start_idx + self.alternate * i for i in range(self.len_snippet) ]
            
        clip_img = []
        img_list = sorted(os.listdir(path_clip))
        # print(len(img_list), path_clip)
        for i in indices:
            img_name = img_list[i]
            # print("img name: ",img_name)
            # print("img name ", img_name)
            img = Image.open(os.path.join(path_clip, img_name)).convert('RGB')
            clip_img.append(self.img_transform(img))
        clip = torch.stack(clip_img, 0).permute(1, 0, 2, 3) #데이터의 맨 앞(0번)에 차원 추가: 스니펫의 길이(16)를 추가한다.
        clip_img = torch.FloatTensor(clip) #Float 형태로 변환
        

        def get_center_slice(arr, length):
            center = len(arr) // 2  # 获取数组的中心位置索引
            start = center - length // 2  # 计算起始索引
            end = start + length  # 计算结束索引
            return arr[start:end]

        # 加载gt maps需要特别关注它的数据分布
        clip_gt = None
        if self.mode != "save" and self.mode != "test":
            print(self.mode)
            gt_sequence_list = get_center_slice(indices, self.gt_length)
            gt_maps = self.get_multi_gt_maps2(gt_sequence_list, path_annt) # (len(seq), 224, 384)
            clip_gt = gt_maps

        data['rgb'] = clip_img
        data["video_id"] = img_name
        data["video_index"] = file_name     # 用于预测
        # print("id, index : ", data["video_id"], data["video_index"])
        if self.mode != "test":
            data["gt_index"] = torch.tensor(gt_sequence_list)


        if self.mode == 'val':
            target['salmap'] = clip_gt
        elif self.mode == "test":
            target['salmap'] = 0
        else:
            target['salmap'] = clip_gt


        return data, target

if __name__ == '__main__':
    
    train_data = MEmoRDataset(path_data="VideoSalPrediction/memor",
                     len_snippet=32, #len_snippet: 한 번에 처리할 프레임의 개수
                     mode="test") #수정이 필요할 듯?

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
        print("test!!!!!!!!!!!!!!",batch["video_id"], batch["gt_index"])