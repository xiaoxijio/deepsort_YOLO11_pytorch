deepsort训练好的模型的夸克下载链接：https://pan.quark.cn/s/c7355fe56ae3  
大家可以放在自己定义的位置  
这是我定义的位置，如果你放别的地方修改default就行  

```
parser.add_argument('--deep_sort_weights', type=str, default='weights/ckpt.t7', help='ckpt.t7 path')
```
说明一下，链接里的就是ckpt.t7，把它放在自己定义的目录下就行。


如果报错
```
FileNotFoundError: [Errno 2] No such file or directory: 'deep_sort_pytorch/deep_sort/deep/checkpoint/ckpt.t7'
```  
将`deepsort_YOLO11_pytorch/deep_sort_pytorch/configs/deep_sort.yaml`里的`REID_CKPT`改成自己的绝对路径，例如
```
REID_CKPT: "/home/tong/workspace/tracker/deepsort_YOLO11_pytorch/weights/ckpt.t7"
```
