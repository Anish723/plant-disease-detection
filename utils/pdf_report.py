from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter
import os


def generate_prediction_pdf(predictions, filepath):

    styles = getSampleStyleSheet()

    story = []

    story.append(Paragraph("Plant Disease Detection Report", styles['Title']))
    story.append(Spacer(1,20))

    for pred in predictions:

        story.append(Paragraph(f"Disease: {pred.result}", styles['Heading3']))
        story.append(Paragraph(f"Confidence: {pred.confidence}%", styles['Normal']))
        story.append(Spacer(1,10))

        img_path = os.path.join("static", pred.image_path)

        if os.path.exists(img_path):

            story.append(Image(img_path, width=200, height=200))

        story.append(Spacer(1,20))

    doc = SimpleDocTemplate(filepath, pagesize=letter)

    doc.build(story)