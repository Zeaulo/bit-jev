"""不依赖 PyTorch 的 BitNet 分隔符和结构化请求编码。"""

import re

from .hub import load_tokenizer


# 五个预留 token 依次标记共享内容、问题、候选项、候选项结束和决策位置。
SPECIAL_TOKENS = ["<|reserved_special_token_0|>", "<|reserved_special_token_1|>",
                  "<|reserved_special_token_2|>", "<|reserved_special_token_3|>",
                  "<|reserved_special_token_4|>"]
DELIMITER_HINTS = dict(zip(SPECIAL_TOKENS, (128002, 128003, 128004, 128005, 128008)))
# 默认训练长度与服务长度按原有模型契约保留。
MAX_STATE, MAX_BRANCH, MAX_PACKED = 384, 1024, 2048
SERVE_MAX_STATE, SERVE_MAX_BRANCH = 8192, 8192
SERVE_MAX_PACKED = SERVE_MAX_STATE + SERVE_MAX_BRANCH
OPT_NONE, OPT_DECIDE = -1, -2
_SPECIAL_RE = re.compile(r"<\|([A-Za-z0-9_]+)\|>")


class ContextOverflow(ValueError):
    """输入内容超出 state、单问题分支或完整请求的长度限制。"""


def pad_id(tokenizer):
    """选择 tokenizer 的填充 token；缺失时沿用零号 token。"""
    return tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0


def delimiter_ids(tokenizer):
    """按 state、question、option、option-end、decide 返回控制 token ID。"""
    ids = []
    for token in SPECIAL_TOKENS:
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id is None or token_id < 0:
            token_id = DELIMITER_HINTS[token]
        if tokenizer.convert_ids_to_tokens(token_id) != token:
            raise ValueError(f"tokenizer 缺少 BitNet 预留 token：{token}")
        ids.append(token_id)
    return ids


def user_tokens(tokenizer, text):
    """转义用户正文中的控制标记，再执行普通文本分词。"""
    return tokenizer(_SPECIAL_RE.sub(r"<¦\1¦>", text), add_special_tokens=False).input_ids


def encode(tokenizer, record, max_state=MAX_STATE, max_branch=MAX_BRANCH, strict=False):
    """把共享 state 与各个问题编码成带候选项边界的 token 序列。"""
    state_tokens = user_tokens(tokenizer, record["state"])
    if strict and len(state_tokens) + 1 > max_state:
        raise ContextOverflow(f"state 超出 {max_state} tokens：{len(state_tokens) + 1}")
    state = [delimiter_ids(tokenizer)[0]] + state_tokens[:max_state - 1]
    ids, seg, pos, opt = list(state), [0] * len(state), list(range(len(state))), [OPT_NONE] * len(state)
    _, question_id, option_id, close_id, decide_id = delimiter_ids(tokenizer)
    decide_positions, option_positions = [], []
    # 每道题形成独立分支；原生路径在调用时再拆成单题因果行。
    for branch_index, question in enumerate(record["questions"], start=1):
        instruction = [question_id] + user_tokens(tokenizer, question["instr"])
        spans = [[option_id] + user_tokens(tokenizer, text) + [close_id]
                 for text in question["options"]]
        branch = instruction + [token for span in spans for token in span] + [decide_id]
        if len(branch) > max_branch - len(state):
            raise ContextOverflow(f"分支超长：{len(branch)} tokens；state 为 {len(state)} tokens")
        base = len(ids)
        ids += branch
        seg += [branch_index] * len(branch)
        pos += list(range(len(state), len(state) + len(branch)))
        opt += [OPT_NONE] * len(instruction) + [index for index, span in enumerate(spans)
                                                    for _ in span] + [OPT_DECIDE]
        ends, cursor = [], len(instruction)
        for span in spans:
            cursor += len(span)
            ends.append(cursor - 1)
        decide_positions.append(base + len(branch) - 1)
        option_positions.append([base + end for end in ends])
    return {"ids": ids, "seg": seg, "pos": pos, "opt": opt,
            "decide_idx": decide_positions, "opt_idx": option_positions,
            "labels": [question["label"] for question in record["questions"]],
            "state_truncated": len(state_tokens) + 1 > max_state}


def fits(record, tokenizer, max_state=MAX_STATE, max_branch=MAX_BRANCH,
         max_packed=MAX_PACKED):
    """检查请求是否能在指定上下文大小内被完整编码。"""
    try:
        return len(encode(tokenizer, record, max_state=max_state,
                          max_branch=max_branch, strict=True)["ids"]) <= max_packed
    except ValueError:
        return False
