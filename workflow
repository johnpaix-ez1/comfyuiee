{
  "id": "18fa08f5-ea86-42c7-93e6-7539622ad7d0",
  "revision": 0,
  "last_node_id": 330,
  "last_link_id": 694,
  "nodes": [
    {
      "id": 253,
      "type": "Label (rgthree)",
      "pos": [
        -1937.36083984375,
        -492.195556640625
      ],
      "size": [
        2399.755859375,
        100
      ],
      "flags": {
        "pinned": true,
        "allow_interaction": false
      },
      "order": 0,
      "mode": 0,
      "inputs": [],
      "outputs": [],
      "title": "Wan 2.1 Skip Layer Guidance - Video2Video Enhance",
      "properties": {
        "fontSize": 100,
        "fontFamily": "Arial",
        "fontColor": "#ffffff",
        "textAlign": "left",
        "backgroundColor": "transparent",
        "padding": 0,
        "borderRadius": 0
      },
      "color": "#fff0",
      "bgcolor": "#fff0"
    },
    {
      "id": 255,
      "type": "Label (rgthree)",
      "pos": [
        -1926.9371337890625,
        -359.23175048828125
      ],
      "size": [
        1660.810546875,
        60
      ],
      "flags": {
        "pinned": true,
        "allow_interaction": false
      },
      "order": 1,
      "mode": 0,
      "inputs": [],
      "outputs": [],
      "title": "Video Tutorial , Go To : https://www.patreon.com/c/aifuturetech",
      "properties": {
        "fontSize": 60,
        "fontFamily": "Arial",
        "fontColor": "#ffffff",
        "textAlign": "left",
        "backgroundColor": "transparent",
        "padding": 0,
        "borderRadius": 0
      },
      "color": "#fff0",
      "bgcolor": "#fff0"
    },
    {
      "id": 221,
      "type": "WanVideoTeaCacheKJ",
      "pos": [
        485.9472351074219,
        69.36123657226562
      ],
      "size": [
        315,
        154
      ],
      "flags": {},
      "order": 13,
      "mode": 0,
      "inputs": [
        {
          "label": "model",
          "name": "model",
          "type": "MODEL",
          "link": 473
        }
      ],
      "outputs": [
        {
          "label": "model",
          "name": "model",
          "type": "MODEL",
          "slot_index": 0,
          "links": [
            578
          ]
        }
      ],
      "properties": {
        "cnr_id": "comfyui-kjnodes",
        "ver": "c0f9894dc5b9dbc35db3e4679bf05f0d4b3d59c0",
        "Node name for S&R": "WanVideoTeaCacheKJ"
      },
      "widgets_values": [
        0.03,
        0.2,
        1,
        "offload_device",
        "disabled"
      ]
    },
    {
      "id": 219,
      "type": "CLIPTextEncode",
      "pos": [
        57.234649658203125,
        453.8829345703125
      ],
      "size": [
        425.27801513671875,
        180.6060791015625
      ],
      "flags": {},
      "order": 8,
      "mode": 0,
      "inputs": [
        {
          "label": "clip",
          "name": "clip",
          "type": "CLIP",
          "link": 585
        }
      ],
      "outputs": [
        {
          "label": "CONDITIONING",
          "name": "CONDITIONING",
          "type": "CONDITIONING",
          "slot_index": 0,
          "links": [
            598
          ]
        }
      ],
      "title": "CLIP Text Encode (Negative Prompt)",
      "properties": {
        "cnr_id": "comfy-core",
        "ver": "0.3.26",
        "Node name for S&R": "CLIPTextEncode"
      },
      "widgets_values": [
        "Overexposure, static, blurred details, subtitles, paintings, pictures, still, overall gray, worst quality, low quality, JPEG compression residue, ugly, mutilated, redundant fingers, poorly painted hands, poorly painted faces, deformed, disfigured, deformed limbs, fused fingers, cluttered background, three legs, a lot of people in the background, upside down"
      ],
      "color": "#322",
      "bgcolor": "#533"
    },
    {
      "id": 252,
      "type": "LayerUtility: PurgeVRAM",
      "pos": [
        566.934814453125,
        629.5868530273438
      ],
      "size": [
        315,
        82
      ],
      "flags": {},
      "order": 28,
      "mode": 0,
      "inputs": [
        {
          "name": "anything",
          "type": "*",
          "link": 520
        }
      ],
      "outputs": [],
      "properties": {
        "cnr_id": "comfyui_layerstyle",
        "ver": "4b273d4f08ea28b0743ababab70ae2e6d93be194",
        "Node name for S&R": "LayerUtility: PurgeVRAM"
      },
      "widgets_values": [
        true,
        true
      ],
      "color": "rgba(38, 73, 116, 0.7)"
    },
    {
      "id": 271,
      "type": "ImageScaleBy",
      "pos": [
        2575.321533203125,
        154.28759765625
      ],
      "size": [
        315,
        82
      ],
      "flags": {},
      "order": 30,
      "mode": 4,
      "inputs": [
        {
          "name": "image",
          "type": "IMAGE",
          "link": 590
        }
      ],
      "outputs": [
        {
          "name": "IMAGE",
          "type": "IMAGE",
          "slot_index": 0,
          "links": [
            561,
            568
          ]
        }
      ],
      "properties": {
        "cnr_id": "comfy-core",
        "ver": "0.3.26",
        "Node name for S&R": "ImageScaleBy"
      },
      "widgets_values": [
        "bicubic",
        2
      ]
    },
    {
      "id": 280,
      "type": "VHS_VideoCombine",
      "pos": [
        2463.498291015625,
        491.7318420410156
      ],
      "size": [
        920.9451293945312,
        334
      ],
      "flags": {
        "collapsed": false
      },
      "order": 31,
      "mode": 4,
      "inputs": [
        {
          "label": "images",
          "name": "images",
          "type": "IMAGE",
          "link": 561
        },
        {
          "label": "audio",
          "name": "audio",
          "shape": 7,
          "type": "AUDIO",
          "link": null
        },
        {
          "label": "meta_batch",
          "name": "meta_batch",
          "shape": 7,
          "type": "VHS_BatchManager",
          "link": null
        },
        {
          "label": "vae",
          "name": "vae",
          "shape": 7,
          "type": "VAE",
          "link": null
        },
        {
          "label": "frame_rate",
          "name": "frame_rate",
          "type": "FLOAT",
          "widget": {
            "name": "frame_rate"
          },
          "link": 592
        }
      ],
      "outputs": [
        {
          "label": "Filenames",
          "name": "Filenames",
          "type": "VHS_FILENAMES"
        }
      ],
      "properties": {
        "cnr_id": "comfyui-videohelpersuite",
        "ver": "f4b4764694abe8002a2c1601f56403c1a7128bdc",
        "Node name for S&R": "VHS_VideoCombine"
      },
      "widgets_values": {
        "frame_rate": 16,
        "loop_count": 0,
        "filename_prefix": "WanVideo",
        "format": "video/h264-mp4",
        "pix_fmt": "yuv420p",
        "crf": 19,
        "save_metadata": true,
        "trim_to_audio": false,
        "pingpong": false,
        "save_output": false,
        "videopreview": {
          "paused": true,
          "hidden": true,
          "params": {
            "filename": "WanVideo_00021.mp4",
            "workflow": "WanVideo_00021.png",
            "fullpath": "C:\\Video\\ComfyUI\\temp\\WanVideo_00021.mp4",
            "format": "video/h264-mp4",
            "subfolder": "",
            "type": "temp",
            "frame_rate": 24
          }
        }
      }
    },
    {
      "id": 281,
      "type": "VHS_VideoCombine",
      "pos": [
        3469.423583984375,
        480.534423828125
      ],
      "size": [
        1343.84619140625,
        334
      ],
      "flags": {
        "collapsed": false
      },
      "order": 33,
      "mode": 4,
      "inputs": [
        {
          "label": "images",
          "name": "images",
          "type": "IMAGE",
          "link": 569
        },
        {
          "label": "audio",
          "name": "audio",
          "shape": 7,
          "type": "AUDIO",
          "link": null
        },
        {
          "label": "meta_batch",
          "name": "meta_batch",
          "shape": 7,
          "type": "VHS_BatchManager",
          "link": null
        },
        {
          "label": "vae",
          "name": "vae",
          "shape": 7,
          "type": "VAE",
          "link": null
        },
        {
          "label": "frame_rate",
          "name": "frame_rate",
          "type": "FLOAT",
          "widget": {
            "name": "frame_rate"
          },
          "link": 593
        }
      ],
      "outputs": [
        {
          "label": "Filenames",
          "name": "Filenames",
          "type": "VHS_FILENAMES"
        }
      ],
      "properties": {
        "cnr_id": "comfyui-videohelpersuite",
        "ver": "f4b4764694abe8002a2c1601f56403c1a7128bdc",
        "Node name for S&R": "VHS_VideoCombine"
      },
      "widgets_values": {
        "frame_rate": 16,
        "loop_count": 0,
        "filename_prefix": "WanVideo",
        "format": "video/h264-mp4",
        "pix_fmt": "yuv420p",
        "crf": 19,
        "save_metadata": true,
        "trim_to_audio": false,
        "pingpong": false,
        "save_output": false,
        "videopreview": {
          "paused": true,
          "hidden": true,
          "params": {
            "filename": "WanVideo_00023.mp4",
            "workflow": "WanVideo_00023.png",
            "fullpath": "C:\\Video\\ComfyUI\\temp\\WanVideo_00023.mp4",
            "format": "video/h264-mp4",
            "subfolder": "",
            "type": "temp",
            "frame_rate": 24
          }
        }
      }
    },
    {
      "id": 291,
      "type": "ImageBlur",
      "pos": [
        -820.5968627929688,
        888.7645263671875
      ],
      "size": [
        315,
        82
      ],
      "flags": {},
      "order": 19,
      "mode": 4,
      "inputs": [
        {
          "label": "image",
          "name": "image",
          "type": "IMAGE",
          "link": 602
        }
      ],
      "outputs": [
        {
          "label": "IMAGE",
          "name": "IMAGE",
          "type": "IMAGE",
          "slot_index": 0,
          "links": [
            595
          ]
        }
      ],
      "properties": {
        "cnr_id": "comfy-core",
        "ver": "0.3.26",
        "Node name for S&R": "ImageBlur"
      },
      "widgets_values": [
        13,
        0.30000000000000004
      ]
    },
    {
      "id": 287,
      "type": "LoraLoaderModelOnly",
      "pos": [
        -122.2723617553711,
        893.86376953125
      ],
      "size": [
        445.9898376464844,
        82
      ],
      "flags": {},
      "order": 16,
      "mode": 4,
      "inputs": [
        {
          "label": "model",
          "name": "model",
          "type": "MODEL",
          "link": 578
        }
      ],
      "outputs": [
        {
          "label": "MODEL",
          "name": "MODEL",
          "type": "MODEL",
          "slot_index": 0,
          "links": [
            579
          ]
        }
      ],
      "properties": {
        "cnr_id": "comfy-core",
        "ver": "0.3.26",
        "Node name for S&R": "LoraLoaderModelOnly"
      },
      "widgets_values": [
        "wan2.1-1.3b-control-lora-tile-v0.2_comfy.safetensors",
        1
      ]
    },
    {
      "id": 292,
      "type": "InstructPixToPixConditioning",
      "pos": [
        -429.24273681640625,
        891.985595703125
      ],
      "size": [
        235.1999969482422,
        86
      ],
      "flags": {},
      "order": 24,
      "mode": 4,
      "inputs": [
        {
          "label": "positive",
          "name": "positive",
          "type": "CONDITIONING",
          "link": 640
        },
        {
          "label": "negative",
          "name": "negative",
          "type": "CONDITIONING",
          "link": 598
        },
        {
          "label": "vae",
          "name": "vae",
          "type": "VAE",
          "link": 596
        },
        {
          "label": "pixels",
          "name": "pixels",
          "type": "IMAGE",
          "link": 595
        }
      ],
      "outputs": [
        {
          "label": "positive",
          "name": "positive",
          "type": "CONDITIONING",
          "slot_index": 0,
          "links": [
            599
          ]
        },
        {
          "label": "negative",
          "name": "negative",
          "type": "CONDITIONING",
          "slot_index": 1,
          "links": [
            600
          ]
        },
        {
          "label": "latent",
          "name": "latent",
          "type": "LATENT",
          "slot_index": 2,
          "links": [
            608
          ]
        }
      ],
      "properties": {
        "cnr_id": "comfy-core",
        "ver": "0.3.26",
        "Node name for S&R": "InstructPixToPixConditioning"
      },
      "widgets_values": []
    },
    {
      "id": 284,
      "type": "CR Upscale Image",
      "pos": [
        2986.798095703125,
        126.45067596435547
      ],
      "size": [
        315,
        222
      ],
      "flags": {},
      "order": 32,
      "mode": 4,
      "inputs": [
        {
          "name": "image",
          "type": "IMAGE",
          "link": 568
        }
      ],
      "outputs": [
        {
          "name": "IMAGE",
          "type": "IMAGE",
          "slot_index": 0,
          "links": [
            569
          ]
        },
        {
          "name": "show_help",
          "type": "STRING",
          "links": null
        }
      ],
      "properties": {
        "cnr_id": "ComfyUI_Comfyroll_CustomNodes",
        "ver": "d78b780ae43fcf8c6b7c6505e6ffb4584281ceca",
        "Node name for S&R": "CR Upscale Image"
      },
      "widgets_values": [
        "4x-UltraSharp.pth",
        "rescale",
        2,
        1024,
        "lanczos",
        "true",
        8
      ]
    },
    {
      "id": 227,
      "type": "VAEEncodeTiled",
      "pos": [
        638.1251220703125,
        819.3208618164062
      ],
      "size": [
        315,
        150
      ],
      "flags": {},
      "order": 18,
      "mode": 0,
      "inputs": [
        {
          "label": "pixels",
          "name": "pixels",
          "type": "IMAGE",
          "link": 589
        },
        {
          "label": "vae",
          "name": "vae",
          "type": "VAE",
          "link": 601
        }
      ],
      "outputs": [
        {
          "label": "LATENT",
          "name": "LATENT",
          "type": "LATENT",
          "slot_index": 0,
          "links": [
            609
          ]
        }
      ],
      "properties": {
        "cnr_id": "comfy-core",
        "ver": "0.3.26",
        "Node name for S&R": "VAEEncodeTiled"
      },
      "widgets_values": [
        256,
        64,
        64,
        8
      ]
    },
    {
      "id": 293,
      "type": "Switch latent [Crystools]",
      "pos": [
        181.34104919433594,
        1049.7410888671875
      ],
      "size": [
        315,
        78
      ],
      "flags": {},
      "order": 25,
      "mode": 4,
      "inputs": [
        {
          "name": "on_true",
          "type": "LATENT",
          "link": 609
        },
        {
          "name": "on_false",
          "type": "LATENT",
          "link": 608
        }
      ],
      "outputs": [
        {
          "name": "latent",
          "type": "LATENT",
          "slot_index": 0,
          "links": [
            607
          ]
        }
      ],
      "properties": {
        "cnr_id": "comfyui-crystools",
        "ver": "1.22.0",
        "Node name for S&R": "Switch latent [Crystools]"
      },
      "widgets_values": [
        false
      ]
    },
    {
      "id": 228,
      "type": "VAEDecodeTiled",
      "pos": [
        890.2109375,
        551.263427734375
      ],
      "size": [
        315,
        150
      ],
      "flags": {},
      "order": 27,
      "mode": 0,
      "inputs": [
        {
          "name": "samples",
          "type": "LATENT",
          "link": 486
        },
        {
          "name": "vae",
          "type": "VAE",
          "link": 587
        }
      ],
      "outputs": [
        {
          "name": "IMAGE",
          "type": "IMAGE",
          "slot_index": 0,
          "links": [
            520,
            521,
            590
          ]
        }
      ],
      "properties": {
        "cnr_id": "comfy-core",
        "ver": "0.3.26",
        "Node name for S&R": "VAEDecodeTiled"
      },
      "widgets_values": [
        512,
        64,
        64,
        8
      ]
    },
    {
      "id": 290,
      "type": "Fast Groups Bypasser (rgthree)",
      "pos": [
        640.42333984375,
        1073.7943115234375
      ],
      "size": [
        456.9739990234375,
        123.56919860839844
      ],
      "flags": {},
      "order": 2,
      "mode": 0,
      "inputs": [],
      "outputs": [
        {
          "name": "OPT_CONNECTION",
          "type": "*",
          "slot_index": 0,
          "links": null
        }
      ],
      "properties": {
        "matchColors": "",
        "matchTitle": "",
        "showNav": true,
        "sort": "position",
        "customSortAlphabet": "",
        "toggleRestriction": "default"
      }
    },
    {
      "id": 248,
      "type": "VHS_VideoCombine",
      "pos": [
        1274.1878662109375,
        78.56008911132812
      ],
      "size": [
        1119.3538818359375,
        2259.919677734375
      ],
      "flags": {
        "collapsed": false
      },
      "order": 29,
      "mode": 0,
      "inputs": [
        {
          "label": "images",
          "name": "images",
          "type": "IMAGE",
          "link": 521
        },
        {
          "label": "audio",
          "name": "audio",
          "shape": 7,
          "type": "AUDIO",
          "link": 613
        },
        {
          "label": "meta_batch",
          "name": "meta_batch",
          "shape": 7,
          "type": "VHS_BatchManager",
          "link": null
        },
        {
          "label": "vae",
          "name": "vae",
          "shape": 7,
          "type": "VAE",
          "link": null
        },
        {
          "label": "frame_rate",
          "name": "frame_rate",
          "type": "FLOAT",
          "widget": {
            "name": "frame_rate"
          },
          "link": 594
        }
      ],
      "outputs": [
        {
          "label": "Filenames",
          "name": "Filenames",
          "type": "VHS_FILENAMES"
        }
      ],
      "properties": {
        "cnr_id": "comfyui-videohelpersuite",
        "ver": "f4b4764694abe8002a2c1601f56403c1a7128bdc",
        "Node name for S&R": "VHS_VideoCombine"
      },
      "widgets_values": {
        "frame_rate": 16,
        "loop_count": 0,
        "filename_prefix": "WanVideo",
        "format": "video/h264-mp4",
        "pix_fmt": "yuv420p",
        "crf": 19,
        "save_metadata": true,
        "trim_to_audio": false,
        "pingpong": false,
        "save_output": true,
        "videopreview": {
          "paused": false,
          "hidden": false,
          "params": {
            "filename": "WanVideo_00009-audio.mp4",
            "workflow": "WanVideo_00009.png",
            "fullpath": "/root/ComfyUI/output/WanVideo_00009-audio.mp4",
            "format": "video/h264-mp4",
            "subfolder": "",
            "type": "output",
            "frame_rate": 16
          }
        }
      }
    },
    {
      "id": 211,
      "type": "ImageResizeKJ",
      "pos": [
        -1324.6246337890625,
        215.4613494873047
      ],
      "size": [
        315,
        286
      ],
      "flags": {},
      "order": 15,
      "mode": 0,
      "inputs": [
        {
          "name": "image",
          "type": "IMAGE",
          "link": 523
        },
        {
          "name": "get_image_size",
          "shape": 7,
          "type": "IMAGE",
          "link": null
        },
        {
          "name": "width",
          "type": "INT",
          "widget": {
            "name": "width"
          },
          "link": 610
        },
        {
          "name": "height",
          "type": "INT",
          "widget": {
            "name": "height"
          },
          "link": 611
        }
      ],
      "outputs": [
        {
          "name": "IMAGE",
          "type": "IMAGE",
          "slot_index": 0,
          "links": [
            589,
            602
          ]
        },
        {
          "name": "width",
          "type": "INT",
          "links": null
        },
        {
          "name": "height",
          "type": "INT",
          "links": null
        }
      ],
      "properties": {
        "cnr_id": "comfyui-kjnodes",
        "ver": "a5bd3c86c8ed6b83c55c2d0e7a59515b15a0137f",
        "Node name for S&R": "ImageResizeKJ"
      },
      "widgets_values": [
        848,
        480,
        "bilinear",
        false,
        2,
        0
      ]
    },
    {
      "id": 288,
      "type": "VHS_VideoInfo",
      "pos": [
        -1280.367431640625,
        720.0087890625
      ],
      "size": [
        262,
        206
      ],
      "flags": {},
      "order": 12,
      "mode": 0,
      "inputs": [
        {
          "name": "video_info",
          "type": "VHS_VIDEOINFO",
          "link": 591
        }
      ],
      "outputs": [
        {
          "name": "source_fps🟨",
          "type": "FLOAT",
          "slot_index": 0,
          "links": [
            592,
            593,
            594
          ]
        },
        {
          "name": "source_frame_count🟨",
          "type": "INT",
          "links": null
        },
        {
          "name": "source_duration🟨",
          "type": "FLOAT",
          "links": null
        },
        {
          "name": "source_width🟨",
          "type": "INT",
          "links": [
            610
          ]
        },
        {
          "name": "source_height🟨",
          "type": "INT",
          "links": [
            611
          ]
        },
        {
          "name": "loaded_fps🟦",
          "type": "FLOAT",
          "links": null
        },
        {
          "name": "loaded_frame_count🟦",
          "type": "INT",
          "links": null
        },
        {
          "name": "loaded_duration🟦",
          "type": "FLOAT",
          "links": null
        },
        {
          "name": "loaded_width🟦",
          "type": "INT",
          "links": null
        },
        {
          "name": "loaded_height🟦",
          "type": "INT",
          "links": null
        }
      ],
      "properties": {
        "cnr_id": "comfyui-videohelpersuite",
        "ver": "5e61bcf218fe3bb7c899bbd584bbc99a9d05fb42",
        "Node name for S&R": "VHS_VideoInfo"
      },
      "widgets_values": {}
    },
    {
      "id": 294,
      "type": "Reroute",
      "pos": [
        -307.54534912109375,
        -64.92387390136719
      ],
      "size": [
        75,
        26
      ],
      "flags": {},
      "order": 11,
      "mode": 0,
      "inputs": [
        {
          "name": "",
          "type": "*",
          "link": 612
        }
      ],
      "outputs": [
        {
          "name": "",
          "type": "AUDIO",
          "links": [
            613
          ]
        }
      ],
      "properties": {
        "showOutputText": false,
        "horizontal": false
      }
    },
    {
      "id": 38,
      "type": "CLIPLoader",
      "pos": [
        -460.5172119140625,
        367.3461608886719
      ],
      "size": [
        370,
        106
      ],
      "flags": {},
      "order": 3,
      "mode": 0,
      "inputs": [],
      "outputs": [
        {
          "label": "CLIP",
          "name": "CLIP",
          "type": "CLIP",
          "slot_index": 0,
          "links": [
            584,
            585
          ]
        }
      ],
      "properties": {
        "cnr_id": "comfy-core",
        "ver": "0.3.26",
        "Node name for S&R": "CLIPLoader"
      },
      "widgets_values": [
        "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
        "wan",
        "default"
      ],
      "color": "#432",
      "bgcolor": "#653"
    },
    {
      "id": 39,
      "type": "VAELoader",
      "pos": [
        -445.058837890625,
        515.2313842773438
      ],
      "size": [
        320,
        58
      ],
      "flags": {},
      "order": 4,
      "mode": 0,
      "inputs": [],
      "outputs": [
        {
          "label": "VAE",
          "name": "VAE",
          "type": "VAE",
          "slot_index": 0,
          "links": [
            587,
            596,
            601
          ]
        }
      ],
      "properties": {
        "cnr_id": "comfy-core",
        "ver": "0.3.26",
        "Node name for S&R": "VAELoader"
      },
      "widgets_values": [
        "wan_2.1_vae.safetensors"
      ],
      "color": "#332922",
      "bgcolor": "#593930"
    },
    {
      "id": 220,
      "type": "CLIPTextEncode",
      "pos": [
        98.23092651367188,
        277.2950134277344
      ],
      "size": [
        384.6632080078125,
        88
      ],
      "flags": {},
      "order": 22,
      "mode": 0,
      "inputs": [
        {
          "label": "clip",
          "name": "clip",
          "type": "CLIP",
          "link": 584
        },
        {
          "label": "text",
          "name": "text",
          "type": "STRING",
          "widget": {
            "name": "text"
          },
          "link": 693
        }
      ],
      "outputs": [
        {
          "label": "CONDITIONING",
          "name": "CONDITIONING",
          "type": "CONDITIONING",
          "slot_index": 0,
          "links": [
            640
          ]
        }
      ],
      "title": "CLIP Text Encode (Positive Prompt)",
      "properties": {
        "cnr_id": "comfy-core",
        "ver": "0.3.26",
        "Node name for S&R": "CLIPTextEncode"
      },
      "widgets_values": [
        "a beautiful girl"
      ],
      "color": "#232",
      "bgcolor": "#353"
    },
    {
      "id": 37,
      "type": "UNETLoader",
      "pos": [
        -493.585205078125,
        32.24568557739258
      ],
      "size": [
        410.84185791015625,
        85.9513168334961
      ],
      "flags": {},
      "order": 5,
      "mode": 0,
      "inputs": [],
      "outputs": [
        {
          "label": "MODEL",
          "name": "MODEL",
          "type": "MODEL",
          "slot_index": 0,
          "links": [
            583
          ]
        }
      ],
      "properties": {
        "cnr_id": "comfy-core",
        "ver": "0.3.26",
        "Node name for S&R": "UNETLoader"
      },
      "widgets_values": [
        "wan2.1_t2v_1.3b_fp16.safetensors",
        "default"
      ]
    },
    {
      "id": 286,
      "type": "VHS_LoadVideo",
      "pos": [
        -2041.276611328125,
        100.48575592041016
      ],
      "size": [
        469.37469482421875,
        1105.521240234375
      ],
      "flags": {},
      "order": 6,
      "mode": 0,
      "inputs": [
        {
          "name": "meta_batch",
          "shape": 7,
          "type": "VHS_BatchManager",
          "link": null
        },
        {
          "name": "vae",
          "shape": 7,
          "type": "VAE",
          "link": null
        }
      ],
      "outputs": [
        {
          "name": "IMAGE",
          "type": "IMAGE",
          "slot_index": 0,
          "links": [
            576
          ]
        },
        {
          "name": "frame_count",
          "type": "INT",
          "links": null
        },
        {
          "name": "audio",
          "type": "AUDIO",
          "links": [
            612
          ]
        },
        {
          "name": "video_info",
          "type": "VHS_VIDEOINFO",
          "slot_index": 3,
          "links": [
            591
          ]
        }
      ],
      "properties": {
        "cnr_id": "comfyui-videohelpersuite",
        "ver": "5e61bcf218fe3bb7c899bbd584bbc99a9d05fb42",
        "Node name for S&R": "VHS_LoadVideo"
      },
      "widgets_values": {
        "video": "MMaudio_00011-audio.mp4",
        "force_rate": 0,
        "custom_width": 0,
        "custom_height": 0,
        "frame_load_cap": 0,
        "skip_first_frames": 0,
        "select_every_nth": 1,
        "format": "AnimateDiff",
        "choose video to upload": "image",
        "videopreview": {
          "hidden": false,
          "paused": false,
          "params": {
            "force_rate": 0,
            "custom_width": 0,
            "custom_height": 0,
            "frame_load_cap": 0,
            "skip_first_frames": 0,
            "select_every_nth": 1,
            "filename": "MMaudio_00011-audio.mp4",
            "type": "input",
            "format": "video/mp4"
          }
        }
      }
    },
    {
      "id": 223,
      "type": "SkipLayerGuidanceWanVideo",
      "pos": [
        51.31256866455078,
        113.68824768066406
      ],
      "size": [
        352.79998779296875,
        106
      ],
      "flags": {},
      "order": 9,
      "mode": 0,
      "inputs": [
        {
          "name": "model",
          "type": "MODEL",
          "link": 583
        }
      ],
      "outputs": [
        {
          "name": "MODEL",
          "type": "MODEL",
          "slot_index": 0,
          "links": [
            473
          ]
        }
      ],
      "properties": {
        "cnr_id": "comfyui-kjnodes",
        "ver": "7c488a16ef420acf0276a4f8e31fc024a969d24b",
        "Node name for S&R": "SkipLayerGuidanceWanVideo"
      },
      "widgets_values": [
        "9,10",
        0.10000000000000002,
        0.9500000000000002
      ]
    },
    {
      "id": 256,
      "type": "Reroute",
      "pos": [
        -1451.1690673828125,
        332.5063171386719
      ],
      "size": [
        75,
        26
      ],
      "flags": {},
      "order": 10,
      "mode": 0,
      "inputs": [
        {
          "name": "",
          "type": "*",
          "link": 576
        }
      ],
      "outputs": [
        {
          "name": "",
          "type": "IMAGE",
          "slot_index": 0,
          "links": [
            523,
            524,
            614
          ]
        }
      ],
      "properties": {
        "showOutputText": false,
        "horizontal": false
      }
    },
    {
      "id": 226,
      "type": "KSampler",
      "pos": [
        882.2864379882812,
        167.945068359375
      ],
      "size": [
        315,
        262
      ],
      "flags": {},
      "order": 26,
      "mode": 0,
      "inputs": [
        {
          "label": "model",
          "name": "model",
          "type": "MODEL",
          "link": 579
        },
        {
          "label": "positive",
          "name": "positive",
          "type": "CONDITIONING",
          "link": 599
        },
        {
          "label": "negative",
          "name": "negative",
          "type": "CONDITIONING",
          "link": 600
        },
        {
          "label": "latent_image",
          "name": "latent_image",
          "type": "LATENT",
          "link": 607
        }
      ],
      "outputs": [
        {
          "label": "LATENT",
          "name": "LATENT",
          "type": "LATENT",
          "slot_index": 0,
          "links": [
            486
          ]
        }
      ],
      "properties": {
        "cnr_id": "comfy-core",
        "ver": "0.3.26",
        "Node name for S&R": "KSampler"
      },
      "widgets_values": [
        132044810230253,
        "randomize",
        20,
        5,
        "uni_pc",
        "simple",
        0.4000000000000001
      ]
    },
    {
      "id": 295,
      "type": "GetImageSizeAndCount",
      "pos": [
        -1572.6990966796875,
        624.3512573242188
      ],
      "size": [
        190.86483764648438,
        86
      ],
      "flags": {},
      "order": 14,
      "mode": 0,
      "inputs": [
        {
          "name": "image",
          "type": "IMAGE",
          "link": 614
        }
      ],
      "outputs": [
        {
          "name": "image",
          "type": "IMAGE",
          "links": null
        },
        {
          "label": "368 width",
          "name": "width",
          "type": "INT",
          "links": null
        },
        {
          "label": "640 height",
          "name": "height",
          "type": "INT",
          "links": null
        },
        {
          "label": "97 count",
          "name": "count",
          "type": "INT",
          "links": [
            694
          ]
        }
      ],
      "properties": {
        "cnr_id": "comfyui-kjnodes",
        "ver": "1.1.2",
        "Node name for S&R": "GetImageSizeAndCount"
      },
      "widgets_values": []
    },
    {
      "id": 209,
      "type": "ImageFromBatch+",
      "pos": [
        -1290.5194091796875,
        561.5611572265625
      ],
      "size": [
        315,
        82
      ],
      "flags": {},
      "order": 17,
      "mode": 0,
      "inputs": [
        {
          "name": "image",
          "type": "IMAGE",
          "link": 524
        },
        {
          "name": "length",
          "type": "INT",
          "widget": {
            "name": "length"
          },
          "link": 694
        }
      ],
      "outputs": [
        {
          "name": "IMAGE",
          "type": "IMAGE",
          "slot_index": 0,
          "links": [
            458
          ]
        }
      ],
      "properties": {
        "cnr_id": "comfyui_essentials",
        "ver": "1.1.0",
        "Node name for S&R": "ImageFromBatch+"
      },
      "widgets_values": [
        0,
        1
      ]
    },
    {
      "id": 210,
      "type": "DownloadAndLoadFlorence2Model",
      "pos": [
        -1310.571044921875,
        30.854894638061523
      ],
      "size": [
        365.4000244140625,
        130
      ],
      "flags": {},
      "order": 7,
      "mode": 0,
      "inputs": [
        {
          "name": "lora",
          "shape": 7,
          "type": "PEFTLORA",
          "link": null
        }
      ],
      "outputs": [
        {
          "name": "florence2_model",
          "type": "FL2MODEL",
          "links": [
            459
          ]
        }
      ],
      "properties": {
        "cnr_id": "comfyui-florence2",
        "ver": "c9bd1d34eb8689746366d4bb34dfbb195aa8d0e1",
        "Node name for S&R": "DownloadAndLoadFlorence2Model"
      },
      "widgets_values": [
        "MiaoshouAI/Florence-2-base-PromptGen-v2.0",
        "fp16",
        "sdpa",
        false
      ]
    },
    {
      "id": 188,
      "type": "Florence2Run",
      "pos": [
        -937.0428466796875,
        240.8281707763672
      ],
      "size": [
        400,
        364
      ],
      "flags": {},
      "order": 20,
      "mode": 0,
      "inputs": [
        {
          "label": "image",
          "name": "image",
          "type": "IMAGE",
          "link": 458
        },
        {
          "label": "florence2_model",
          "name": "florence2_model",
          "type": "FL2MODEL",
          "link": 459
        }
      ],
      "outputs": [
        {
          "label": "image",
          "name": "image",
          "type": "IMAGE"
        },
        {
          "label": "mask",
          "name": "mask",
          "type": "MASK"
        },
        {
          "label": "caption",
          "name": "caption",
          "type": "STRING",
          "slot_index": 2,
          "links": [
            665,
            693
          ]
        },
        {
          "label": "data",
          "name": "data",
          "type": "JSON"
        }
      ],
      "properties": {
        "cnr_id": "comfyui-florence2",
        "ver": "f59d5c34d9c5693473e873dacbedd86857749cf4",
        "Node name for S&R": "Florence2Run"
      },
      "widgets_values": [
        "",
        "more_detailed_caption",
        true,
        false,
        4096,
        3,
        true,
        "",
        77777777777778,
        "fixed"
      ]
    },
    {
      "id": 320,
      "type": "Bjornulf_DisplayNote",
      "pos": [
        -365.4273986816406,
        194.98501586914062
      ],
      "size": [
        236.79998779296875,
        88
      ],
      "flags": {},
      "order": 21,
      "mode": 0,
      "inputs": [
        {
          "name": "any",
          "type": "*",
          "link": 665
        }
      ],
      "outputs": [
        {
          "name": "any",
          "type": "*",
          "links": [
            691
          ]
        }
      ],
      "properties": {
        "cnr_id": "bjornulf_custom_nodes",
        "ver": "1.1.8",
        "Node name for S&R": "Bjornulf_DisplayNote"
      },
      "widgets_values": [
        ""
      ]
    },
    {
      "id": 329,
      "type": "PreviewAny",
      "pos": [
        -105.36531066894531,
        208.82644653320312
      ],
      "size": [
        140,
        76.00001525878906
      ],
      "flags": {},
      "order": 23,
      "mode": 0,
      "inputs": [
        {
          "name": "source",
          "type": "*",
          "link": 691
        }
      ],
      "outputs": [],
      "properties": {
        "cnr_id": "comfy-core",
        "ver": "0.3.43",
        "Node name for S&R": "PreviewAny"
      },
      "widgets_values": []
    }
  ],
  "links": [
    [
      458,
      209,
      0,
      188,
      0,
      "IMAGE"
    ],
    [
      459,
      210,
      0,
      188,
      1,
      "FL2MODEL"
    ],
    [
      473,
      223,
      0,
      221,
      0,
      "MODEL"
    ],
    [
      486,
      226,
      0,
      228,
      0,
      "LATENT"
    ],
    [
      520,
      228,
      0,
      252,
      0,
      "*"
    ],
    [
      521,
      228,
      0,
      248,
      0,
      "IMAGE"
    ],
    [
      523,
      256,
      0,
      211,
      0,
      "IMAGE"
    ],
    [
      524,
      256,
      0,
      209,
      0,
      "IMAGE"
    ],
    [
      561,
      271,
      0,
      280,
      0,
      "IMAGE"
    ],
    [
      568,
      271,
      0,
      284,
      0,
      "IMAGE"
    ],
    [
      569,
      284,
      0,
      281,
      0,
      "IMAGE"
    ],
    [
      576,
      286,
      0,
      256,
      0,
      "*"
    ],
    [
      578,
      221,
      0,
      287,
      0,
      "MODEL"
    ],
    [
      579,
      287,
      0,
      226,
      0,
      "MODEL"
    ],
    [
      583,
      37,
      0,
      223,
      0,
      "MODEL"
    ],
    [
      584,
      38,
      0,
      220,
      0,
      "CLIP"
    ],
    [
      585,
      38,
      0,
      219,
      0,
      "CLIP"
    ],
    [
      587,
      39,
      0,
      228,
      1,
      "VAE"
    ],
    [
      589,
      211,
      0,
      227,
      0,
      "IMAGE"
    ],
    [
      590,
      228,
      0,
      271,
      0,
      "IMAGE"
    ],
    [
      591,
      286,
      3,
      288,
      0,
      "VHS_VIDEOINFO"
    ],
    [
      592,
      288,
      0,
      280,
      4,
      "*"
    ],
    [
      593,
      288,
      0,
      281,
      4,
      "*"
    ],
    [
      594,
      288,
      0,
      248,
      4,
      "*"
    ],
    [
      595,
      291,
      0,
      292,
      3,
      "IMAGE"
    ],
    [
      596,
      39,
      0,
      292,
      2,
      "VAE"
    ],
    [
      598,
      219,
      0,
      292,
      1,
      "CONDITIONING"
    ],
    [
      599,
      292,
      0,
      226,
      1,
      "CONDITIONING"
    ],
    [
      600,
      292,
      1,
      226,
      2,
      "CONDITIONING"
    ],
    [
      601,
      39,
      0,
      227,
      1,
      "VAE"
    ],
    [
      602,
      211,
      0,
      291,
      0,
      "IMAGE"
    ],
    [
      607,
      293,
      0,
      226,
      3,
      "LATENT"
    ],
    [
      608,
      292,
      2,
      293,
      1,
      "LATENT"
    ],
    [
      609,
      227,
      0,
      293,
      0,
      "LATENT"
    ],
    [
      610,
      288,
      3,
      211,
      2,
      "INT"
    ],
    [
      611,
      288,
      4,
      211,
      3,
      "INT"
    ],
    [
      612,
      286,
      2,
      294,
      0,
      "*"
    ],
    [
      613,
      294,
      0,
      248,
      1,
      "AUDIO"
    ],
    [
      614,
      256,
      0,
      295,
      0,
      "IMAGE"
    ],
    [
      640,
      220,
      0,
      292,
      0,
      "CONDITIONING"
    ],
    [
      665,
      188,
      2,
      320,
      0,
      "*"
    ],
    [
      691,
      320,
      0,
      329,
      0,
      "*"
    ],
    [
      693,
      188,
      2,
      220,
      1,
      "STRING"
    ],
    [
      694,
      295,
      3,
      209,
      1,
      "INT"
    ]
  ],
  "groups": [
    {
      "id": 7,
      "title": "Upscale",
      "bounding": [
        2438.95556640625,
        39.802528381347656,
        2259.779052734375,
        1040.367919921875
      ],
      "color": "#3f789e",
      "font_size": 24,
      "flags": {}
    },
    {
      "id": 8,
      "title": "With Tile Lora",
      "bounding": [
        -864.06884765625,
        742.8348999023438,
        1388.4381103515625,
        460.08038330078125
      ],
      "color": "#3f789e",
      "font_size": 24,
      "flags": {}
    }
  ],
  "config": {},
  "extra": {
    "ds": {
      "scale": 0.6115909044841478,
      "offset": [
        2004.6519942552904,
        78.96393443444164
      ]
    },
    "VHS_KeepIntermediate": true,
    "VHS_MetadataImage": true,
    "0246.VERSION": [
      0,
      0,
      4
    ],
    "VHS_latentpreviewrate": 0,
    "VHS_latentpreview": false,
    "node_versions": {
      "comfy-core": "0.3.26",
      "ComfyUI-Florence2": "dffd12506d50f0540b8a7f4b36a05d4fb5fed2de",
      "ComfyUI-KJNodes": "a5bd3c86c8ed6b83c55c2d0e7a59515b15a0137f",
      "ComfyUI-VideoHelperSuite": "c47b10ca1798b4925ff5a5f07d80c51ca80a837d",
      "ComfyUI_LayerStyle": "f8439eb17f03e0fa60a35303493bfc9a7d5ab098",
      "ComfyUI_essentials": "33ff89fd354d8ec3ab6affb605a79a931b445d99",
      "ComfyUI_Comfyroll_CustomNodes": "d78b780ae43fcf8c6b7c6505e6ffb4584281ceca"
    },
    "ue_links": [],
    "frontendVersion": "1.20.7"
  },
  "version": 0.4
}
