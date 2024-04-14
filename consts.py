
Denoise_Bilateral_Mode = "Anime4K_Denoise_Bilateral_Mode.glsl"
Denoise_Bilateral_Median = "Anime4K_Denoise_Bilateral_Median.glsl"

DarkLines_VeryFast = "Anime4K_Darken_VeryFast.glsl"
DarkLines_Fast = "Anime4K_Darken_Fast.glsl"
DarkLines_HQ = "Anime4K_Darken_HQ.glsl"

ThinLines_VeryFast = "Anime4K_Thin_VeryFast.glsl"
ThinLines_Fast = "Anime4K_Thin_Fast.glsl"
ThinLines_HQ = "Anime4K_Thin_HQ.glsl"

Deblur_DoG = "Anime4K_Deblur_DoG.glsl"

Upscale_CNN_M_x2_Deblur = "Anime4K_Deblur_Original.glsl"
Upscale_CNN_L_x2_Deblur = "Anime4K_Deblur_Original.glsl"
Upscale_CNN_UL_x2_Deblur = "Anime4K_Deblur_Original.glsll"

Upscale_CNN_M_x2_Denoise = "Anime4K_Upscale_Denoise_CNN_x2_M.glsl"
Upscale_CNN_L_x2_Denoise = "Anime4K_Upscale_Denoise_CNN_x2_L.glsl"
Upscale_CNN_UL_x2_Denoise = "Anime4K_Upscale_Denoise_CNN_x2_UL.glsl"


Auto_Downscale_Pre_x4 = "Anime4K_AutoDownscalePre_x4.glsl"

IS_GUI = False
GUI_OPTS = {
    "upscale": {
        "width": 1920,
        "height": 1080,
        "cg_choice": 0,
        "shader_mode_choice": 2,
        "shader_quality_choice": 2,
        "shader_bilateral_choice": 1,
        "x264_preset": "medium",
        "x264_lossless": 0
    },
    "encode": {
        "mode": 0
    },
    "audio": {

    }
}