// bit-jev 原生 CPU 推理：逐题运行 BitNet，并在候选边界读取隐藏状态。
#include "llama.h"
#include "nlohmann/json.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

using Json = nlohmann::json;

// 模型加载时仅输出警告和错误，避免每次启动打印整份张量目录。
void native_log(ggml_log_level level, const char * message, void *) {
    if (level >= GGML_LOG_LEVEL_WARN) std::cerr << message;
}

// 指针头文件的矩阵均按行优先排列，顺序与 Python 导出器一致。
struct Head {
    uint32_t hidden_size = 0;
    uint32_t head_size = 0;
    float temperature = 1.0f;
    std::vector<float> q_weight;
    std::vector<float> q_bias;
    std::vector<float> k_weight;
    std::vector<float> k_bias;
};

// llama_batch 持有 C API 分配的内存，在异常路径也自动释放。
struct BatchHolder {
    explicit BatchHolder(int capacity) : batch(llama_batch_init(capacity, 0, 1)) {}
    ~BatchHolder() { llama_batch_free(batch); }
    llama_batch batch;
};

// 从固定布局的 sidecar 读取全部指针头参数。
Head load_head(const std::string & path) {
    std::ifstream source(path, std::ios::binary);
    if (!source) throw std::runtime_error("无法打开 head.f32：" + path);
    char magic[8] = {};
    Head head;
    source.read(magic, 8);
    source.read(reinterpret_cast<char *>(&head.hidden_size), sizeof(head.hidden_size));
    source.read(reinterpret_cast<char *>(&head.head_size), sizeof(head.head_size));
    source.read(reinterpret_cast<char *>(&head.temperature), sizeof(head.temperature));
    if (!source || std::memcmp(magic, "BJHEAD01", 8) != 0 || head.hidden_size == 0 ||
        head.head_size == 0 || !std::isfinite(head.temperature) || head.temperature <= 0) {
        throw std::runtime_error("head.f32 文件头无效");
    }
    // 四段数组的尺寸由文件头决定，读取长度不足即视为损坏。
    const size_t matrix_size = static_cast<size_t>(head.hidden_size) * head.head_size;
    auto read_array = [&source](size_t count) {
        std::vector<float> values(count);
        source.read(reinterpret_cast<char *>(values.data()), count * sizeof(float));
        if (!source) throw std::runtime_error("head.f32 参数数据截断");
        return values;
    };
    head.q_weight = read_array(matrix_size);
    head.q_bias = read_array(head.head_size);
    head.k_weight = read_array(matrix_size);
    head.k_bias = read_array(head.head_size);
    if (source.peek() != std::char_traits<char>::eof()) {
        throw std::runtime_error("head.f32 含有多余数据");
    }
    return head;
}

// 将一个骨干隐藏状态投影到指针空间。
std::vector<float> project(const std::vector<float> & hidden,
                           const std::vector<float> & weights,
                           const std::vector<float> & bias,
                           const Head & head) {
    std::vector<float> result(head.head_size);
    for (size_t row = 0; row < head.head_size; ++row) {
        float value = bias[row];
        const size_t offset = row * head.hidden_size;
        for (size_t col = 0; col < head.hidden_size; ++col) {
            value += weights[offset + col] * hidden[col];
        }
        result[row] = value;
    }
    return result;
}

// 清空上一题的 KV 缓存后，按普通因果顺序运行一行。
std::vector<std::vector<float>> read_hidden(llama_context * context, const Json & row,
                                             uint32_t hidden_size, int batch_size) {
    const std::vector<llama_token> ids = row.at("ids").get<std::vector<llama_token>>();
    const std::vector<int> option_positions = row.at("options").get<std::vector<int>>();
    const int decide_position = row.at("decide").get<int>();
    if (ids.empty() || option_positions.empty() || decide_position != static_cast<int>(ids.size()) - 1 ||
        ids.size() > 4096) {
        throw std::runtime_error("题目 token 序列或 decide 位置无效");
    }
    // 每个输出位置都应位于当前因果行内且互不重复。
    std::unordered_set<int> wanted(option_positions.begin(), option_positions.end());
    wanted.insert(decide_position);
    if (wanted.size() != option_positions.size() + 1) {
        throw std::runtime_error("候选结束位置重复或与 decide 重叠");
    }
    for (int position : wanted) {
        if (position < 0 || position >= static_cast<int>(ids.size())) {
            throw std::runtime_error("隐藏状态读取位置越界");
        }
    }
    llama_memory_clear(llama_get_memory(context), true);
    BatchHolder holder(batch_size);
    std::vector<std::vector<float>> hidden(ids.size());
    // 分块送入模型；每次 decode 后立即复制本块要求的隐藏状态。
    for (size_t start = 0; start < ids.size(); start += batch_size) {
        const int count = static_cast<int>(std::min<size_t>(batch_size, ids.size() - start));
        holder.batch.n_tokens = count;
        for (int index = 0; index < count; ++index) {
            const int position = static_cast<int>(start) + index;
            holder.batch.token[index] = ids[position];
            holder.batch.pos[index] = position;
            holder.batch.n_seq_id[index] = 1;
            holder.batch.seq_id[index][0] = 0;
            // embeddings 模式要求当前块每个 token 都为输出；后续仅复制需要的位置。
            holder.batch.logits[index] = 1;
        }
        if (llama_decode(context, holder.batch) != 0) {
            throw std::runtime_error("BitNet CPU decode 失败");
        }
        for (int index = 0; index < count; ++index) {
            const int position = static_cast<int>(start) + index;
            if (!wanted.count(position)) continue;
            const float * embedding = llama_get_embeddings_ith(context, index);
            if (embedding == nullptr) throw std::runtime_error("BitNet 未返回请求的隐藏状态");
            hidden[position].assign(embedding, embedding + hidden_size);
        }
    }
    // 结果按候选顺序排列，最后一项是 decide 状态。
    std::vector<std::vector<float>> selected;
    selected.reserve(option_positions.size() + 1);
    for (int position : option_positions) selected.push_back(std::move(hidden[position]));
    selected.push_back(std::move(hidden[decide_position]));
    return selected;
}

// 对候选状态执行与 Python PointerHead 完全相同的投影和温度缩放。
std::vector<float> score_row(const std::vector<std::vector<float>> & states, const Head & head) {
    const std::vector<float> query = project(states.back(), head.q_weight, head.q_bias, head);
    const float scale = 1.0f / std::sqrt(static_cast<float>(head.head_size)) / head.temperature;
    std::vector<float> logits;
    logits.reserve(states.size() - 1);
    for (size_t option = 0; option + 1 < states.size(); ++option) {
        const std::vector<float> key = project(states[option], head.k_weight, head.k_bias, head);
        float dot = 0.0f;
        for (size_t index = 0; index < head.head_size; ++index) dot += query[index] * key[index];
        logits.push_back(dot * scale);
    }
    return logits;
}

// 使用稳定的 softmax 将一题的 logits 转换为概率。
std::vector<float> softmax(const std::vector<float> & logits) {
    const float maximum = *std::max_element(logits.begin(), logits.end());
    std::vector<float> probabilities;
    probabilities.reserve(logits.size());
    float sum = 0.0f;
    for (float value : logits) {
        const float probability = std::exp(value - maximum);
        probabilities.push_back(probability);
        sum += probability;
    }
    for (float & probability : probabilities) probability /= sum;
    return probabilities;
}

// 读取命令行选项；所有输入均为显式路径或正整数。
std::string option_value(int argc, char ** argv, const std::string & name,
                         const std::string & fallback = "") {
    for (int index = 1; index + 1 < argc; ++index) {
        if (argv[index] == name) return argv[index + 1];
    }
    return fallback;
}

int main(int argc, char ** argv) {
    const std::string model_path = option_value(argc, argv, "--model");
    const std::string head_path = option_value(argc, argv, "--head");
    const int threads = std::stoi(option_value(argc, argv, "--threads", "4"));
    const int batch_size = std::stoi(option_value(argc, argv, "--batch", "256"));
    if (model_path.empty() || head_path.empty() || threads <= 0 || batch_size <= 0) {
        std::cerr << "用法：bit-jev-cpu --model 文件.gguf --head head.f32 --threads 4 --batch 256\n";
        return 2;
    }
    llama_log_set(native_log, nullptr);
    llama_backend_init();
    try {
        // 主干仅加载到 CPU；嵌入输出来自最后一层的 result_norm。
        llama_model_params model_params = llama_model_default_params();
        model_params.n_gpu_layers = 0;
        std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
            llama_model_load_from_file(model_path.c_str(), model_params), &llama_model_free);
        if (!model) throw std::runtime_error("无法加载 I2_S GGUF 模型");
        const Head head = load_head(head_path);
        if (head.hidden_size != static_cast<uint32_t>(llama_model_n_embd(model.get()))) {
            throw std::runtime_error("指针头与 BitNet 隐藏维度不一致");
        }
        llama_context_params context_params = llama_context_default_params();
        context_params.n_ctx = 4096;
        context_params.n_batch = batch_size;
        context_params.n_ubatch = batch_size;
        context_params.n_outputs_max = batch_size;
        context_params.n_threads = threads;
        context_params.n_threads_batch = threads;
        context_params.embeddings = true;
        context_params.pooling_type = LLAMA_POOLING_TYPE_NONE;
        std::unique_ptr<llama_context, decltype(&llama_free)> context(
            llama_init_from_model(model.get(), context_params), &llama_free);
        if (!context) throw std::runtime_error("无法初始化 BitNet CPU 上下文");
        llama_set_embeddings(context.get(), true);
        // 模型常驻内存；stdin 每一行独立对应 stdout 的一行结果。
        std::string line;
        while (std::getline(std::cin, line)) {
            if (line.empty()) continue;
            const auto started = std::chrono::steady_clock::now();
            const Json request = Json::parse(line);
            Json result;
            result["logits"] = Json::array();
            result["probabilities"] = Json::array();
            for (const Json & row : request.at("rows")) {
                auto states = read_hidden(context.get(), row, head.hidden_size, batch_size);
                auto logits = score_row(states, head);
                result["logits"].push_back(logits);
                result["probabilities"].push_back(softmax(logits));
            }
            const auto ended = std::chrono::steady_clock::now();
            result["latency_ms"] = std::chrono::duration<double, std::milli>(ended - started).count();
            std::cout << result.dump() << '\n' << std::flush;
        }
    } catch (const std::exception & error) {
        std::cerr << "bit-jev CPU 推理失败：" << error.what() << '\n';
        llama_backend_free();
        return 1;
    }
    llama_backend_free();
    return 0;
}
