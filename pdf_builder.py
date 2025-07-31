from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
import matplotlib.pyplot as plt
import numpy as np
import io

def save_training_plot(actual, predicted):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(actual, label='Actual', linewidth=2)
    ax.plot(predicted, label='Predicted', linestyle='--')
    ax.set_title("Training Fit (Last Sequences)")
    ax.legend()
    ax.grid(True)

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    return buf

def add_token_page(c, name, report_text, next_pred, actual, predicted, buf):
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.utils import ImageReader

    width, height = A4
    y = height - 50  # Start from top

    c.setFont("Helvetica-Bold", 18)
    c.drawString(50, y, f"📄 Report — {name}")
    y -= 30

    # Report text block
    c.setFont("Helvetica", 12)
    text = c.beginText(50, y)
    for line in report_text.split("\n"):
        text.textLine(line)
        y -= 15
    c.drawText(text)

    # Prediction info
    y -= 20
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, f"🔮 Next Predicted Price (normalized): {next_pred:.4f}")
    y -= 30

    # First plot (training fit)
    plot_buf = save_training_plot(actual, predicted)
    img1_height = 200
    c.drawImage(ImageReader(plot_buf), 50, y - img1_height, width=500, preserveAspectRatio=True, mask='auto')
    y -= (img1_height + 20)
    c.drawString(50, y, "📊 Training Fit")

    # Second plot (future prediction)
    img2_height = 200
    c.drawImage(ImageReader(buf), 50, y - img2_height - 20, width=500, preserveAspectRatio=True, mask='auto')
    c.drawString(50, y - img2_height - 40, "📈 Future Price Prediction")

    c.showPage()


def create_multi_pdf(token_data_list, filename="FULL_TOKENS_REPORT.pdf"):
    c = canvas.Canvas(filename, pagesize=A4)
    for token_data in token_data_list:
        add_token_page(
            c,
            token_data["name"],
            token_data["report"],
            token_data["next_pred"],
            token_data["actual"],
            token_data["predicted"],
            token_data["buf"]
        )
    c.save()
