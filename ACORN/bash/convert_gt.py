import numpy as np

def get_fvecs_count(filename):
    """从 fvecs 文件自动检测向量数量"""
    fv = np.fromfile(filename, dtype=np.float32)
    if fv.size == 0:
        return 0, 0
    dim = fv.view(np.int32)[0]  # 第一个 int32 是维度
    n = fv.size // (dim + 1)    # 总元素数 / (维度+1) = 向量数。数据格式根据utils.cpp推断的读取方式
    return n, dim

def read_groundtruth_file(filename):
    with open(filename, 'rb') as f:
        npts = int.from_bytes(f.read(4), byteorder='little')
        ndims = int.from_bytes(f.read(4), byteorder='little')
        data = np.frombuffer(f.read(npts * ndims * 4), dtype=np.uint32).reshape((npts, ndims))
        distances = np.frombuffer(f.read(npts * ndims * 4), dtype=np.float32).reshape((npts, ndims))
    
    print("npts:", npts)
    print("ndims:", ndims)
    print("data:", data)
    print("distances:", distances)
    
    return npts, ndims, data, distances

'''
read groundtruth file saved by UNG index, namely, saved as:
void write_gt_file(const std::string& filename, const std::pair<IdxType, float>* gt, uint32_t num_queries, uint32_t K) {
    std::ofstream fout(filename, std::ios::binary);
    fout.write(reinterpret_cast<const char*>(gt), num_queries * K * sizeof(std::pair<IdxType, float>));
    std::cout << "Ground truth written to " << filename << std::endl;
}
'''

def read_ung_gt(filename, num_queries, K):
    with open(filename, 'rb') as f:
        gt_data = np.frombuffer(f.read(num_queries * K * 8), dtype=np.dtype([('idx', np.uint32), ('dist', np.float32)]))
        gt_data = gt_data.reshape((num_queries, K))
    
    # Extract indices and distances
    indices = gt_data['idx']
    distances = gt_data['dist']
    
    print("indices:", indices)
    print("distances:", distances)
    
    return indices, distances

# num_queries = 200
# num_queries = 354

# dataset = 'sift'
dataset = 'words'
scenarios = ['and', 'or']
# scenario = 'NHQ'
# for K in [1, 25, 50, 100]:
K = 10
# for sel in [1, 25, 50, 75]:
for scenario in scenarios:
    query_file = "../../data/" + dataset + "/" + dataset + "_query_" + scenario + ".fvecs"
    num_queries, _ = get_fvecs_count(query_file)
    gt_path = "../../data/" + dataset + "/" + dataset + "_gt_" + scenario + ".bin"
    output_path = "../../data/" + dataset + "/" + dataset + "_gt_" + scenario + ".txt"

    gt, _ = read_ung_gt(gt_path, num_queries, K)

    with open(output_path, 'w') as f:
        for i in range(num_queries):
            f.write(' '.join([str(x) for x in gt[i]]) + '\n')
