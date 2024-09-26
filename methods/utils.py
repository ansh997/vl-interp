def projection(image_embeddings, text_embedding):
    return (image_embeddings @ text_embedding.T)[0, :, 0] / (text_embedding @ text_embedding.T).squeeze()

def subtract_projection(image_embeddings, text_embedding, weight = 1):
    image_embeddings = image_embeddings.clone()
    proj = projection(image_embeddings, text_embedding)
    for i in range(image_embeddings.shape[1]):
        if proj[i] > 0:
            image_embeddings[:, i] -= weight * proj[i] * text_embedding
    return image_embeddings

def subtract_projections(image_embeddings, text_embeddings, weight = 1):
    # text_embeddings: (# embeds, 1, # dim size)
    img_embeddings = image_embeddings.clone()
    for text_embedding in text_embeddings:
        img_embeddings = subtract_projection(img_embeddings, text_embedding, weight)
    return img_embeddings

def remove_all_hooks(model):
    # Iterate over all modules in the model
    for module in model.modules():
        # Clear forward hooks
        if hasattr(module, '_forward_hooks'):
            module._forward_hooks.clear()
        # Clear backward hooks (if any)
        if hasattr(module, '_backward_hooks'):
            module._backward_hooks.clear()
        # Clear forward pre-hooks (if any)
        if hasattr(module, '_forward_pre_hooks'):
            module._forward_pre_hooks.clear()

def generate_mass_edit_hook(text_embeddings, start_edit_index, end_edit_index, layer, weight = 1, minimum_size = 32):
    def edit_embeddings(module, input, output):
        new_output = list(output)
        if new_output[0].shape[1] > minimum_size:
            print(f"Editing layer {layer}")
            new_output[0][:, start_edit_index: end_edit_index] = subtract_projections(new_output[0][:, start_edit_index:end_edit_index], text_embeddings, weight = weight)
        return tuple(new_output)
    return edit_embeddings

def generate_mass_edit_pre_hook(text_embeddings, start_edit_index, end_edit_index, layer, weight = 1, minimum_size = 32):
    def edit_embeddings(module, input):
        new_input = list(input)
        if new_input[0].shape[1] > minimum_size:
            print(f"Editing layer {layer}")
            new_input[0][:, start_edit_index: end_edit_index] = subtract_projections(new_input[0][:, start_edit_index:end_edit_index], text_embeddings, weight = weight)
        return tuple(new_input)
    return edit_embeddings