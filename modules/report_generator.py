from reportlab.platypus import (

    SimpleDocTemplate,

    Paragraph,

    Spacer
)

from reportlab.lib.styles \
    import getSampleStyleSheet

import io

def create_report(
        model_name,
        problem_type,
        metrics,
        training_time,
        cv_score
):

    buffer = io.BytesIO()

    pdf = SimpleDocTemplate(
        buffer
    )

    styles = \
        getSampleStyleSheet()

    content = []
    
    content.append(

        Paragraph(

            "AutoML Studio Report",

            styles["Title"]
        )
    )

    content.append(
        Spacer(1, 20)
    )
    
    content.append(

        Paragraph(

            f"<b>Model:</b> "
            f"{model_name}",

            styles["Normal"]
        )
    )
    
    content.append(

        Paragraph(

            f"<b>Problem Type:</b> "
            f"{problem_type}",

            styles["Normal"]
        )
    )
    
    content.append(

        Paragraph(

            f"<b>Training Time:</b> "
            f"{training_time:.2f}s",

            styles["Normal"]
        )
    )
    
    cv_display = (
            round(cv_score, 4)
            if cv_score is not None
            else "N/A"
        )
    
    content.append(

        Paragraph(

            f"<b>CV Score:</b> "
            f"{cv_display}",

            styles["Normal"]
        )
    ) 
    
    content.append(
        Spacer(1, 20)
    )
    
    content.append(

        Paragraph(

            "Metrics",

            styles["Heading2"]
        )
    )   
    
    for k, v in metrics.items():

        if not isinstance(v, str):

            v = str(v)

        v = v.replace("\n", "<br/>")

        content.append(

            Paragraph(

                f"<b>{k}</b>: {v}",

                styles["Normal"]
            )
        )
        
    pdf.build(content)

    buffer.seek(0)

    return buffer