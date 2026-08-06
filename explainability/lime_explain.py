from lime import lime_image
import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries

def lime_explanation(image, model):
    explainer = lime_image.LimeImageExplainer()

    explanation = explainer.explain_instance(
        image[0].astype('double'),
        model.predict,
        top_labels=1,
        hide_color=0,
        num_samples=1000
    )

    temp, mask = explanation.get_image_and_mask(
        explanation.top_labels[0],
        positive_only=True,
        num_features=5,
        hide_rest=False
    )

    plt.imshow(mark_boundaries(temp, mask))
    plt.axis('off')
    plt.show()
