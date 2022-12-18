import mixbox

palette_pigments = [[30, 40, 50], [186, 60, 65], [30, 255, 10], [220, 230, 0], [186, 70, 50], [15, 38, 187], [35, 0, 230], [220, 210, 200]]
weight = [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 3.09991023e-01,
   6.81996486e-01, 0.00000000e+00, 0.00000000e+00, 8.01249058e-03]

def reconstruct_single_color(palette_pigments, weight):
    #input: 
    # palette_pigements needs to be shape(num_pigement, 3)
    # len(weight) needs to be  num_pigment
    rgb_list = []
    for i in range(len(palette_pigments)):
        color = mixbox.rgb_to_latent(palette_pigments[i])
        rgb_list.append(color)
    z_mix = [0] * mixbox.LATENT_SIZE
    for i in range(len(z_mix)): 
        for j in range(len(weight)):  # mix together:
            z_mix[i] += weight[j]*rgb_list[j][i]

    rgb_mix = mixbox.latent_to_rgb(z_mix)
    return rgb_mix

rgb_mix = reconstruct_single_color(palette_pigments, weight)

print(rgb_mix)

