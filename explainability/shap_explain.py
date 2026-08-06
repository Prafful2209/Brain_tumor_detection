import shap

def shap_explanation(model, background, image):
    explainer = shap.DeepExplainer(model, background)
    shap_values = explainer.shap_values(image)
    shap.image_plot(shap_values, image)
    plt.show()
