import os


def configure_huggingface_endpoint(endpoint: str | None = None) -> str | None:
    """Use hf-mirror by default while allowing an explicit environment override."""
    endpoint = HUGGINGFACE_ENDPOINT if endpoint is None else endpoint
    if endpoint:
        os.environ.setdefault("HF_ENDPOINT", endpoint)
    return os.environ.get("HF_ENDPOINT")


#关键帧选取
KEY_FRAME_DELETE_RATIO = 0.5

#分块大小
CHUNK_SIZE = 30
CHUNK_OVERLAP_SIZE = 15

#读取类型，可选：video、image
READ_TYPE_VIDEO = "video"
READ_TYPE_IMAGE = "image"
READ_TYPE = READ_TYPE_VIDEO
READ_TYPE_CHOICES = (READ_TYPE_VIDEO, READ_TYPE_IMAGE)

#视频读取参数
VIDEO_SEGMENT_SECONDS = 10
VIDEO_EXTRACT_FPS = 3

#图片读取参数
IMAGE_SEGMENT_SIZE = 50
IMAGE_SAMPLE_INTERVAL = 3

#语义分割遮罩
REMOVE_MASK_ENABLE = False
REMOVE_MASK_MODEL_PATH = "nvidia/segformer-b5-finetuned-ade-640-640"
HUGGINGFACE_ENDPOINT = "https://hf-mirror.com"
REMOVE_MASK_MODEL_URL = ""
REMOVE_MASK_SOURCE_PATH = "models/bisenetv2.py"
REMOVE_MASK_SOURCE_URL = "https://raw.githubusercontent.com/CoinCheung/BiSeNet/master/lib/models/bisenetv2.py"
REMOVE_MASK_DEVICE = None
REMOVE_MASK_CLASS_IDS = (2,)
REMOVE_MASK_MEAN = (0.485, 0.456, 0.406)
REMOVE_MASK_STD = (0.229, 0.224, 0.225)

#拼接相关
ALIGN_CONF_THRESHOLD = 0.3      #用于拼接计算的置信度阈值
ALIGN_IRLS_DELTA = 0.1          #鲁棒估计减低某点权重的阈值
ALIGN_IRLS_MAX_ITERS = 5        #鲁棒估计迭代次数
ALIGN_MIN_POINTS = 1            #拼接对齐所需要最少有效点次数
ALIGN_LOOP_ENABLE = True        #是否启用非相邻分块回环约束
ALIGN_LOOP_MIN_CHUNK_GAP = 1    #回环约束要求两个分块至少间隔几个分块
ALIGN_LOOP_MAX_PAIRS = 8        #最多使用多少组回环约束
ALIGN_LOOP_MIN_MATCHES = 20     #回环帧对估计Sim3所需最少特征匹配点数
ALIGN_LOOP_MIN_SCORE = 0.08     #图像特征重叠分数阈值
ALIGN_LOOP_TOP_K = 2            #每个分块最多保留多少个候选回环
ALIGN_LOOP_OPT_MAX_ITERS = 50   #Sim3图优化最大迭代次数
ALIGN_LOOP_WINDOW_RADIUS = 5    #回环共同推理时，以匹配帧为中心向两侧取多少帧
ALIGN_LOOP_MODEL_NAME_OR_PATH = "facebook/VGGT-1B"


#输出参数
EXPORT_GLB_ENABLE = True        #是否导出最终GLB
EXPORT_PLY_ENABLE = True        #是否导出最终PLY
ALIGN_GLB_CONF_THRES = 0.0     #最终GLB输出的置信度百分位阈值
ALIGN_PLY_CONF_THRES = 0.0     #最终PLY输出的置信度百分位阈值
POINTCLOUD_DOWNSAMPLE_ENABLE = False        #是否在最终导出后额外生成下采样点云
POINTCLOUD_DOWNSAMPLE_VOXEL_SIZE = 0.002    #voxel grid体素大小
POINTCLOUD_DOWNSAMPLE_MIN_KEEP_RATIO = 0.5    #下采样后最少保留的点比例
POINTCLOUD_DOWNSAMPLE_SUFFIX = "_downsampled"
RESULT_DIR = "data/result"
