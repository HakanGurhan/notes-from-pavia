"""
Pavia'dan Notlar #01 — Titanic'te hayatta kalır mıydın?
Uses the logistic regression weights trained in the homework notebook
(Machine Learning course, Università degli Studi di Pavia).
"""

import json
import math
import gradio as gr

# ---------------------------------------------------------------------------
# Load the trained logistic regression weights.
# These were learned on the Titanic dataset (710 train / 177 test samples).
# Order of features: [Pclass, Sex, Age, SibSp, Parch, Fare]
# ---------------------------------------------------------------------------
with open("model_weights.json") as f:
    data = json.load(f)

W = data["w"]                         # length-6 list of floats
B = data["b"][0] if isinstance(data["b"], list) else data["b"]


def sigmoid(z: float) -> float:
    """Numerically safe sigmoid."""
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def predict(pclass, sex, age, sibsp, parch, fare):
    """Run the same logistic regression inference as in the notebook:
       p = sigmoid(X · w + b)"""
    # Convert categorical inputs to the same encoding used during training
    sex_val = 1 if sex == "Kadın" else 0
    pclass_val = int(pclass)

    x = [pclass_val, sex_val, float(age),
         int(sibsp), int(parch), float(fare)]

    logit = sum(xi * wi for xi, wi in zip(x, W)) + B
    p = sigmoid(logit) * 100  # as percentage

    # Build a Markdown report
    if p >= 70:
        verdict = "🟢 **Büyük olasılıkla HAYATTA KALIRDIN**"
        color = "#1F4E79"
    elif p >= 40:
        verdict = "🟡 **Şansın 50/50**"
        color = "#B8860B"
    else:
        verdict = "🔴 **Maalesef hayatta kalma şansın düşük**"
        color = "#A6332E"

    # ASCII probability bar
    filled = int(p / 5)
    bar = "█" * filled + "░" * (20 - filled)

    # Per-feature contribution (so users can see *why* their score is what it is)
    contributions = []
    feature_labels = ["Bilet sınıfı", "Cinsiyet", "Yaş",
                      "Kardeş/eş", "Ebeveyn/çocuk", "Bilet fiyatı"]
    for label, xi, wi in zip(feature_labels, x, W):
        contributions.append((label, xi * wi))
    contributions.sort(key=lambda t: -abs(t[1]))

    contrib_lines = "\n".join(
        f"- **{lbl}:** {val:+.2f}" for lbl, val in contributions[:4]
    )

    return (
        f"## Hayatta kalma olasılığın: **{p:.1f}%**\n\n"
        f"`{bar}`\n\n"
        f"{verdict}\n\n"
        f"---\n\n"
        f"**Senin durumunda en etkili 4 faktör:**\n\n"
        f"{contrib_lines}\n\n"
        f"---\n\n"
        f"*Model: Logistic regression, 887 Titanic yolcusu üzerinde eğitildi.*\n"
        f"*Test doğruluğu: %80.79*"
    )


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
with gr.Blocks(
    title="Titanic'te hayatta kalır mıydın?",
    theme=gr.themes.Soft(primary_hue="blue"),
) as demo:

    gr.Markdown(
        """
        # 🚢 Titanic'te hayatta kalır mıydın?

        ### Pavia'dan Notlar #01

        Bu demo, Università degli Studi di Pavia'daki Machine Learning
        dersi için eğittiğim **gerçek logistic regression** modelini
        kullanıyor. 1912 Titanic kazasının verisi üzerinde eğitildi
        (887 yolcu, %80.79 test doğruluğu).

        Bilgilerini gir, modelin sana ne dediğini gör.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Bilgilerin")

            pclass = gr.Radio(
                ["1", "2", "3"], value="3",
                label="Bilet sınıfı",
                info="1 = lüks (üst güverte) • 3 = ekonomi (alt güverte)"
            )
            sex = gr.Radio(
                ["Kadın", "Erkek"], value="Erkek",
                label="Cinsiyet"
            )
            age = gr.Slider(0, 80, value=25, step=1, label="Yaş")
            sibsp = gr.Slider(
                0, 8, value=0, step=1,
                label="Gemideki kardeş veya eş sayısı"
            )
            parch = gr.Slider(
                0, 6, value=0, step=1,
                label="Gemideki ebeveyn veya çocuk sayısı"
            )
            fare = gr.Slider(
                0, 250, value=15, step=1,
                label="Bilet fiyatı (1912 sterlini)",
                info="Referans: 3. sınıf ~£8, 1. sınıf £50-200"
            )

            btn = gr.Button("Kaderimi öğren", variant="primary", size="lg")

        with gr.Column(scale=1):
            out = gr.Markdown(
                "_Bilgilerini girip butona bas..._"
            )

    btn.click(predict,
              inputs=[pclass, sex, age, sibsp, parch, fare],
              outputs=out)

    gr.Examples(
        examples=[
            ["1", "Kadın",  25, 0, 0, 100],  # rich young woman, 1st class
            ["3", "Erkek",  30, 0, 0,   8],  # poor young man, 3rd class
            ["2", "Kadın",   8, 1, 2,  25],  # girl traveling with family
            ["1", "Erkek",  45, 1, 0,  80],  # wealthy man with spouse
        ],
        inputs=[pclass, sex, age, sibsp, parch, fare],
        label="Örnekler — hemen denemek için tıkla",
    )

    gr.Markdown(
        """
        ---

        **Pavia'dan Notlar** — Università degli Studi di Pavia'da aldığım
        yüksek lisans derslerinden öğrendiklerimi paylaştığım seri.
        LinkedIn'de takip et: [#NotesFromPavia](#)
        """
    )

if __name__ == "__main__":
    demo.launch()
