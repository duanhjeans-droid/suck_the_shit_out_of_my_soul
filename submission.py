"""Mẫu bài nộp Wikispeedia — Python 3.10 trở lên.

Hoàn thiện các hàm có TODO; giữ tên hàm, tham số và kiểu đầu ra.
main() đã có phần điều phối và ghi bốn tệp CSV theo đề bài.
Mẫu chưa hoàn thiện sẽ báo NotImplementedError, không sinh đáp án giả.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import math
import hashlib

#Python ơi, dùng UTF-8 để xuất ra cho tớ nhó
import sys
sys.stdout.reconfigure(encoding="utf-8")

# chuyển article thành set -> tránh trùng
# chuyển links thành dict với key = source & value = list các đỉnh đến 
# gọi các link dict bằng các phần tử trong article, nếu có thì lưu list đỉnh đến ở kết quả, nếu không thì để rỗng
# nếu số id ở link thừa số với aritcle, báo value error
def load_graph(
    articles_path: Path, links_path: Path
) -> tuple[list[str], list[list[int]]]:
    """Đọc tập đỉnh và danh sách cạnh từ hai tệp TSV UTF-8.

    Bỏ dòng trống/dòng bắt đầu bằng #. Giữ ID mã hóa phần trăm.
    Trả (ids, out_neighbors): ids là danh sách ID phân biệt sắp theo byte
    UTF-8; out_neighbors[j] chứa các chỉ số đích của đỉnh ids[j].
    Gộp cạnh trùng, giữ cạnh tự nối và mọi đỉnh trong articles.tsv.
    Nút cụt có danh sách đích rỗng. Không gắn cứng số đỉnh/cạnh.
    Báo ValueError nếu đồ thị rỗng hoặc cạnh chứa ID ngoài tập đỉnh.
    """

    articles_set = set()

    with open(articles_path, "r", encoding = "utf-8") as f:
        for line in f:
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            articles_set.add(line)

    if not articles_set:
        raise ValueError("Graph hổng có gì hết trơn (Graph rỗng)")

    articles_set_sorted = sorted(articles_set, key = lambda x: x.encode("utf-8"))

    index = {article_id: i for i, article_id in enumerate(articles_set_sorted)}

    graph = {}
    with open(links_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            block = line.split("\t")

            if len(block)!= 2: 
                raise ValueError(f"{line} - thằng này bị cái gì ýyyyy (links sai format lưu)")

            s, t = block 

            if s not in index or t not in index:
                raise ValueError(f"{line} - thằng này không trong gia phảaaaa (links chứa đối tượng không thuộc articles)")

            if s not in graph: 
                graph[s] = set()
            graph[s].add(t)
    out= [sorted(index[t] for t in graph.get(source,set())) for source in articles_set_sorted]
    return articles_set_sorted, out



def pagerank(
    ids: list[str],
    out_neighbors: list[list[int]],
    beta: float,
    tol: float,
    max_iter: int,
    teleport: list[float] | None = None,
) -> list[float]:
    """Tính PageRank toàn cục hoặc theo phân phối dịch chuyển cho trước.

    ids và out_neighbors dùng cùng chỉ số như kết quả load_graph.
    beta thuộc (0, 1), tol > 0, max_iter là số nguyên >= 1.
    teleport=None dùng phân phối đều; nếu có, teleport phải dài len(ids),
    hữu hạn, không âm và có tổng lệch 1 không quá 1e-12.
    Khởi tạo bằng phân phối dịch chuyển; bù nút cụt luôn đều trên mọi đỉnh.
    Cập nhật đồng thời bằng số thực 64 bit. Dừng khi sai khác L1 giữa hai
    vòng <= tol hoặc đã thực hiện max_iter vòng; không chuẩn hóa lại.
    Trả vector mới nhất theo thứ tự ids, không trả thống kê phụ.
    Báo ValueError nếu tham số hoặc phân phối dịch chuyển không hợp lệ.
    """

    #ktra teleport, khoi tao r0
    if beta <= 0 or beta >= 1:
        raise ValueError(f"Beta ngơ rồi (Invalid beta)")    
    if tol <= 0:
        raise ValueError(f"Tol ngược hướng rồi bé ời (Invalid tol)")
    if max_iter < 1:
        raise ValueError(f"CHo vòng thế này khỏi quay (Invalid max_iter)")
    n = len(ids)

    if teleport is not None:
        if len(teleport) != n or abs(sum(teleport) - 1) >  1e-12:
            raise ValueError(f"Cái thằng teleport bị làm sao ý (Invalid distribution-shifted list)")
        for x in teleport:
            if not isinstance(x, (int, float)) or isinstance(x,bool):
                raise ValueError("teleport chứa giá trị không phải số thực")
            if x <0 or not math.isfinite(x): # bắt âm, NaN, inf
                raise ValueError("teleport chứa giá trị âm hoặc vô hạn")
    else: teleport = [float(1/n)] * n
    
    r = teleport.copy()

    adjacent_list = []
    for i in range (0,n):
        out_list = out_neighbors[i]
        group = (len(out_list), out_list) # vd: (dA, [A, B, D])
        adjacent_list.append(group)

    # tap diem nut cut
    dangling_node = []
    for i in range (0,n):
        if adjacent_list[i][1] == []:
            dangling_node.append(i)
    
    for _ in range( max_iter):
        dangling_point = sum(list(r[x] for x in dangling_node))
        r_t = [0.0] * n
        for i in range (0,n):
            if adjacent_list[i][0]:
                temp = r[i]/adjacent_list[i][0]
                for j in adjacent_list[i][1]:
                    r_t[j] += temp

        for i in range (n): 
            r_t[i] = beta*r_t[i] + dangling_point*beta/n + (1 - beta)*teleport[i]

        diff = sum(abs(a - b) for a, b in zip (r, r_t))
        r = r_t.copy()
        if diff <= tol:
            break
    return r
    



def topic_distribution(
    ids: list[str], categories_path: Path, prefix: str
) -> list[float]:
    """Tạo phân phối dịch chuyển đều trên tập trang thuộc một chủ đề.

    Đọc categories.tsv UTF-8, bỏ dòng trống/dòng bắt đầu bằng #.
    Chọn nhãn bằng prefix hoặc bắt đầu bằng prefix + '.'. Một ID chỉ
    được đếm một lần trong tập chủ đề, dù có nhiều nhãn phù hợp.
    Trả vector dài len(ids), theo thứ tự ids: trang thuộc tập nhận 1/|S|,
    trang còn lại nhận 0. Báo ValueError nếu tập chủ đề rỗng hoặc dữ liệu
    nhãn tham chiếu ID ngoài ids. Không dùng nội dung bài để suy ra nhãn.
    """
    # TODO: Sinh viên cài đặt.
    id_set= set(ids)
    topic_articles= set()
    with open(categories_path, "r", encoding= "utf-8") as f:
        for line in f: 
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts= line.split("\t")
            if len(parts) !=2:
                raise ValueError(f"{line} - thằng này bị cái gì ýyyyy (links sai format lưu)")
            article, label = parts

            # kiem tra id co ton tai khong
            if article not in id_set:
                raise ValueError(f"{line} - thằng này không trong gia phảaaaa (links chứa đối tượng không thuộc articles)")

            # kiem tra nhan co thuoc chu de khong
            if label == prefix or label.startswith(prefix + "."):
                topic_articles.add(article)
    if not topic_articles:
        raise ValueError("Tập chủ đề rỗng")

    # vector phan phối
    weight= 1.0/ len(topic_articles)
    result=[]
    for article in ids:
        if article in topic_articles:
            result.append(weight)
        else:
            result.append(0.0)
    return result


def mix_pageranks(
    topic_scores: list[list[float]], weights: list[float]
) -> list[float]:
    """Ghép các vector PageRank bằng tổng có trọng số theo từng tọa độ.

    topic_scores gồm các vector cùng độ dài, cùng thứ tự đỉnh.
    weights có một trọng số cho mỗi vector, hữu hạn, không âm và có
    tổng lệch 1 không quá 1e-12. Trả một vector có cùng độ dài.
    Cấu hình chính thức dùng thứ tự Science, History, Geography và trọng
    số 0.5, 0.3, 0.2. Không chạy lại PageRank hoặc làm tròn điểm ở đây.
    Báo ValueError nếu danh sách rỗng, kích thước hoặc trọng số không hợp lệ.
    """
    # TODO: Sinh viên cài đặt.
    if not topic_scores:
        raise ValueError(f" topic_score bỏ nhà đi rồi( rỗng)") 
    if not weights:
        raise ValueError(f"weights bỏ nhà đi rồi( rỗng)")
    if len(topic_scores) != len(weights):
        raise ValueError(f" topic_scores và weights dỗi nhau rồi( khác độ dài)")
    for i in range (0,len(topic_scores)):
        if len(topic_scores[0]) != len(topic_scores[i]):
            raise ValueError(f"nội bộ trong topic_scores bất hòa")
    for i in weights:
        if not isinstance(i, (int,float)) or isinstance(i, bool):
            raise ValueError(f" Có giá trị không hợp lệ trong weights")
        if i<0 or not math.isfinite(i):
            raise ValueError(f" Có giá trị âm hoặ vô hạn")
    if abs(sum(weights) - 1.0) > 1e-12:
        raise ValueError(f"Tổng weights phải bằng 1")
    n= len(topic_scores[0])
    results = [0.0] * n
    for vector, w in zip(topic_scores, weights):
        for i in range(len(vector)):
            results[i] += w* vector[i]
    return results

    


def normalize_text(text: str) -> str:
    r"""Chuẩn hóa chuỗi Unicode theo đúng quy tắc của phần C.

    Chỉ đổi ASCII A–Z thành a–z. Gộp mỗi dãy ký tự thuộc tập dấu cách
    ASCII, \t, \n, \r, \v, \f thành một dấu cách ASCII; bỏ dấu cách ASCII
    ở hai đầu. Giữ nguyên các ký tự khác, dấu câu và toàn bộ nội dung.
    Không chuẩn hóa Unicode, bỏ dấu hoặc gọi lower() cho toàn bộ Unicode.
    Trả chuỗi đã chuẩn hóa; chuỗi rỗng hoặc chỉ có khoảng trắng trên trả ''.
    """
    result = []
    in_whitespace = False
    ascii_whitespaces = {' ', '\t', '\n', '\r', '\v', '\f'}
    for char in text :
        if 'A' <= char <= 'Z':
            result.append(chr(ord(char)+32))
            in_whitespace=False
        elif char in ascii_whitespaces:
            if not in_whitespace:
                result.append(' ')
                in_whitespace=True
        else:
            result.append(char)
            in_whitespace=False
    return "".join(result).strip(' ')

    


def shingles(text: str, k: int) -> set[str]:
    """Tạo tập k-shingle từ chuỗi ĐÃ được normalize_text xử lý.

    Mỗi shingle là k code point Unicode liên tiếp, không phải k byte/từ.
    Trả tập chuỗi phân biệt; không lưu số lần xuất hiện hoặc vị trí.
    Chuỗi ngắn hơn k trả tập rỗng. Báo ValueError nếu k không phải số
    nguyên dương. Không chuẩn hóa văn bản thêm lần nữa trong hàm này.
    """
    if not isinstance(k, int) or isinstance(k, bool):
        raise ValueError("k phải là số nguyên dương")
    if k <= 0:
        raise ValueError("Không phải số nguyên dương")
    
    if len(text) < k:
        return set()
    
    k_shingle=set()
    
    for i in range(len(text) - k + 1):
        shingle = text[i:i+k]
        k_shingle.add(shingle)
        
    return k_shingle
    


def shingle_id(shingle: str, p: int) -> int:
    """Ánh xạ một shingle thành số nguyên trong [0, p-1].

    Băm byte UTF-8 bằng SHA-256, lấy 4 byte đầu, đọc số nguyên không dấu
    big-endian rồi lấy modulo p. Không dùng hash() mặc định của Python.
    Trả int tất định giữa các lần chạy. Báo ValueError nếu p <= 1.
    """
    if p <= 1:
        raise ValueError("Giá trị p phải lớn hơn 1.")


    digest = hashlib.sha256(shingle.encode("utf-8")).digest()
    num = int.from_bytes(digest[:4], byteorder="big", signed=False)
    return num % p



def minhash(shingle_ids: set[int], hash_params: dict) -> list[int]:
    """Tính chữ ký từ tập mã shingle và bộ hàm băm dùng chung.

    hash_params có p và hash_functions là danh sách {'a': int, 'b': int}
    theo thứ tự tọa độ. Mỗi tọa độ là min((a*x+b) % p) trên shingle_ids.
    Tập rỗng trả toàn giá trị p. Trả list[int] dài bằng số hàm băm;
    không gắn cứng 128 vì bộ chấm có thể dùng bộ hệ số nhỏ khác.
    Mã đầu vào thuộc [0, p-1]; phép nhân/cộng/modulo dùng số nguyên chính
    xác. Không sinh hệ số mới, không trả ID phần tử đạt min.
    """
    # TODO: Sinh viên cài đặt.
    p = hash_params["p"]
    funcs = hash_params["hash_functions"]
    
    if not shingle_ids:
        return [p] * len(funcs)
    
    # Chuyển set sang list một lần duy nhất để duyệt nhanh hơn
    ids_list = list(shingle_ids)
    
    # Dùng list comprehension để tối ưu tốc độ thay vì vòng lặp for và append
    return [min((h["a"] * x + h["b"]) % p for x in ids_list) for h in funcs]
    raise NotImplementedError("Chưa cài đặt minhash")


def jaccard(set_a: set[str], set_b: set[str]) -> float:
    """Tính Jaccard chính xác trên hai tập CHUỖI shingle trước bước băm.

    Trả kích thước giao chia kích thước hợp. Theo quy ước bài tập, hai tập
    rỗng trả 1.0; chỉ một tập rỗng trả 0.0. Không dùng tập mã băm thay thế.
    """
    # TODO: Sinh viên cài đặt.
    if (not set_a) and (not set_b):
        return 1.0
    if (not set_a) or (not set_b):
        return 0.0
    union_set = set_a | set_b
    intersection_set = set_a & set_b
    return len(intersection_set)/len(union_set)
    raise NotImplementedError("Chưa cài đặt jaccard")


def signature_similarity(
    sig_a: list[int], sig_b: list[int], empty_a: bool, empty_b: bool
) -> float:
    """Ước lượng Jaccard bằng tỷ lệ tọa độ chữ ký trùng nhau.

    Hai chữ ký phải cùng độ dài dương, dùng cùng bộ hàm băm và thứ tự.
    empty_a/empty_b cho biết tập shingle gốc có rỗng hay không.
    Hai tập rỗng trả 1.0; chỉ một tập rỗng trả 0.0. Các trường hợp còn
    lại trả số tọa độ bằng nhau chia độ dài chữ ký. Không chuyển chữ ký
    thành tập. Báo ValueError nếu độ dài không hợp lệ.
    """
    # TODO: Sinh viên cài đặt.
    if (len(sig_a) != len(sig_b)) or len(sig_a) == 0:
        raise ValueError("Độ dài chữ kí không hợp lệ")
    if empty_a and empty_b:
        return 1.0
    if empty_a or empty_b:
        return 0.0
    length = len(sig_a)
    count = 0
    for i in range(0,length):
        if sig_a[i]==sig_b[i]:
            count+=1
    return count/length
    raise NotImplementedError("Chưa cài đặt signature_similarity")


def main() -> None:
    """Đọc tham số, gọi các hàm sinh viên và ghi bốn tệp kết quả CSV.

    Chạy từ thư mục gốc bài tập sau khi giải nén dữ liệu vào data/.
    Dùng --help để xem tùy chọn. Import tệp không tự chạy xử lý dữ liệu.
    Khi các hàm TODO đã hoàn thiện, main tạo pagerank.csv,
    topic_pagerank.csv, minhash.csv và similarities.csv trong --output-dir.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument(
        "--hash-params", type=Path, default=Path("assignment/minhash_params.json")
    )
    parser.add_argument(
        "--pairs", type=Path, default=Path("data/pairs.csv"),
        help="CSV có header article_id_a,article_id_b; mặc định dùng bộ cặp Wikispeedia chính thức.",
    )
    parser.add_argument("--beta", type=float, default=0.85)
    parser.add_argument("--tol", type=float, default=1e-10)
    parser.add_argument("--max-iter", type=int, default=1000)
    args = parser.parse_args()

    graph_dir = args.data_dir / "wikispeedia_paths-and-graph"
    text_dir = args.data_dir / "plaintext_articles"
    with args.hash_params.open(encoding="utf-8") as handle:
        hash_params = json.load(handle)
    with args.pairs.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["article_id_a", "article_id_b"]:
            raise ValueError("Tệp cặp cần header article_id_a,article_id_b")
        pairs = [(row["article_id_a"], row["article_id_b"]) for row in reader]

    ids, out_neighbors = load_graph(
        graph_dir / "articles.tsv", graph_dir / "links.tsv"
    )
    required_ids = {article for pair in pairs for article in pair}
    if not required_ids.issubset(set(ids)):
        raise ValueError("Tệp cặp chứa ID không có trong articles.tsv")

    # A. Điểm PageRank toàn cục.
    scores = pagerank(ids, out_neighbors, args.beta, args.tol, args.max_iter)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "pagerank.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["article_id", "score"])
        for i, article in enumerate(ids):
            writer.writerow([article, format(scores[i], ".17g")])

    # B. Ba chủ đề và r_mix trong cùng một tệp.
    topics = ["Science", "History", "Geography"]
    topic_scores = []
    for topic in topics:
        teleport = topic_distribution(
            ids, graph_dir / "categories.tsv", "subject." + topic
        )
        topic_scores.append(
            pagerank(ids, out_neighbors, args.beta, args.tol, args.max_iter, teleport)
        )
    mixed_scores = mix_pageranks(topic_scores, [0.5, 0.3, 0.2])
    with (args.output_dir / "topic_pagerank.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["topic", "article_id", "score"])
        for topic, vector in zip(topics + ["r_mix"], topic_scores + [mixed_scores]):
            for i, article in enumerate(ids):
                writer.writerow([topic, article, format(vector[i], ".17g")])

    # C. Chữ ký cho mọi bài; chỉ giữ tập chuỗi cho các cặp cần so sánh.
    signatures = {}
    pair_shingles = {}
    with (args.output_dir / "minhash.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle)
        length = len(hash_params["hash_functions"])
        writer.writerow(
            ["article_id", "shingle_count"] + [f"h{j:03d}" for j in range(length)]
        )
        for article in ids:
            text = (text_dir / (article + ".txt")).read_text(encoding="utf-8")
            parts = shingles(normalize_text(text), hash_params["shingle_length"])
            codes = {shingle_id(part, hash_params["p"]) for part in parts}
            signature = minhash(codes, hash_params)
            signatures[article] = signature
            if article in required_ids:
                pair_shingles[article] = parts
            writer.writerow([article, len(parts), *signature])

    with (args.output_dir / "similarities.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["article_id_a", "article_id_b", "jaccard_exact", "jaccard_estimate"]
        )
        for article_a, article_b in pairs:
            set_a, set_b = pair_shingles[article_a], pair_shingles[article_b]
            exact = jaccard(set_a, set_b)
            estimate = signature_similarity(
                signatures[article_a], signatures[article_b], not set_a, not set_b
            )
            writer.writerow(
                [article_a, article_b, format(exact, ".17g"), format(estimate, ".17g")]
            )

    print(f"Đã ghi bốn tệp kết quả vào {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
