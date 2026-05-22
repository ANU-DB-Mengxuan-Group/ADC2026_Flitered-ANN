#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <sstream>
#include <fstream>
#include <faiss/IndexACORN.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexHNSW.h>
#include <faiss/index_io.h>
#include <iomanip>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include "utils.cpp"

int main(int argc, char* argv[]) {
    double t0 = elapsed();

    size_t d = 128; // dimension of the vectors to index
    int M, M_beta, gamma;
    std::string dataset, filename, output_path;
    size_t N = 0;

    // Parse command-line arguments
    if (argc != 8) {
        std::cerr << "Usage: " << argv[0] << " <N> <gamma> <filename> <M> <M_beta> <output_path> <dataset>" << std::endl;
        return 1;
    }

    N = strtoul(argv[1], NULL, 10);
    gamma = atoi(argv[2]);
    filename = argv[3];
    M = atoi(argv[4]);
    M_beta = atoi(argv[5]);
    output_path = argv[6];
    dataset = argv[7];

    // Create indices
    printf("[%.3f s] Index Params -- M: %d, M_beta: %d, N: %ld, gamma: %d\n",
           elapsed() - t0, M, M_beta, N, gamma);

    int efs = 48;

    size_t nb, d2;
    float* xb = fvecs_read(filename.c_str(), &d2, &nb);
    d = d2;

    std::vector<int> meta(N);
    faiss::IndexACORNFlat hybrid_index(d, M, gamma, meta, M_beta);
    hybrid_index.acorn.efSearch = efs;

    double t1 = elapsed();
    hybrid_index.add(N, xb);
    double t2 = elapsed();
    std::cout << "Create gamma index in time: " << t2 - t1 << std::endl;

    delete[] xb;

    // Write indices to files
    {
        std::stringstream filepath_stream;
        filepath_stream << output_path << "/" << dataset << "/hybrid_M=" << M
                        << "_Mb=" << M_beta << "_gamma=" << gamma << ".json";
        std::string filepath = filepath_stream.str();
        write_index(&hybrid_index, filepath.c_str());

        // 计算并保存构建时间到日志之外的文件（用于作为metrics的一部分）
        double construction_time = t2 - t1;  // 这里定义变量
        // 构建时间文件路径：../data/construction_times/{dataset}/M={M}_Mb={M_beta}_gamma={gamma}.time
        std::stringstream time_file_stream;
        time_file_stream << "../data/construction_times/" << dataset 
                        << "/M=" << M << "_Mb=" << M_beta 
                        << "_gamma=" << gamma << ".time";
        std::string time_file = time_file_stream.str();
        
        // 创建目录（提取目录路径）
        std::string time_dir = time_file.substr(0, time_file.find_last_of('/'));
        std::string mkdir_cmd = "mkdir -p " + time_dir;
        system(mkdir_cmd.c_str());
        
        std::ofstream time_out(time_file);
        if (time_out.is_open()) {
            time_out << std::fixed << std::setprecision(6) << construction_time << std::endl;
            time_out.close();
            std::cout << "Construction time saved to: " << time_file << std::endl;
        } else {
            std::cerr << "Warning: Could not write construction time to " << time_file << std::endl;
        }
    }

    std::cout << std::endl;
    return 0;
}