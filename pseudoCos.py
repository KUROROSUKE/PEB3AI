import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

element_weights = {
    "H": 0.16666,
    "C": 0.25,
    "O": 0.25
    # 他の元素は 1.0 のまま
}

# 🎯 **ファイルを読み込む**
with open("compound/obf_extended_min.json", "r", encoding="utf-8") as file:
    data = json.load(file)

# 🎯 **元素リストを取得**
all_elements = set()
for material in data["material"]:
    all_elements.update(material["d"].keys())
all_elements = sorted(all_elements)  # 一貫した順序でソート

# 🎯 **物質をベクトル化**
def convert_to_vector(composition):
    return np.array([
        composition.get(el, 0) * element_weights.get(el, 1.0)
        for el in all_elements
    ])

vectors = np.array([convert_to_vector(m["d"]) for m in data["material"]])

# 🎯 **コサイン類似度を計算**
similarity_matrix = cosine_similarity(vectors)

# 🎯 **最も近い物質を見つける（閾値 0.5 以下なら g を null）**
THRESHOLD = 0.5

for i, material in enumerate(data["material"]):
    similarities = similarity_matrix[i]
    sorted_indices = np.argsort(similarities)[::-1]  # 類似度が高い順
    closest_material = None

    for j in sorted_indices:
        if j != i:  # **自分自身を除く**
            if similarities[j] > THRESHOLD:
                closest_material = data["material"][j]["f"]
            break  # **最も近い物質を 1 つだけ取得**

    material["g"] = closest_material  # **閾値以下なら `null` に設定**

# 🎯 **新しい JSON を保存**
output_path = "compound/obs_standard_with_g_threshold.json"
with open(output_path, "w", encoding="utf-8") as file:
    json.dump(data, file, ensure_ascii=False, indent=4)

print(f"処理が完了しました。新しい JSON ファイル: {output_path}")
