# Test images

Sample photos for trying the app. File names give the correct answer.

| Folder | Source | What they look like |
|---|---|---|
| `plantvillage/` | [PlantVillage](https://github.com/spMohanty/PlantVillage-Dataset) | Single leaf, plain background, lab lighting |
| `plantdoc/` | [PlantDoc](https://github.com/pratikkayal/PlantDoc-Dataset) (test split) | Real field / internet photos |

Results from the trained model (`ResNet50V2_plantdoc`), plant left as "I don't know":

| Image | Prediction | Confidence |
|---|---|---|
| plantvillage/apple_scab.jpg | Apple scab | 100% |
| plantvillage/corn_common_rust.jpg | Corn common rust | 100% |
| plantvillage/grape_black_rot.jpg | Grape black rot | 99% |
| plantvillage/tomato_healthy.jpg | Tomato healthy | 100% |
| plantdoc/corn_common_rust.jpg | Corn common rust | 98% |
| plantdoc/apple_cedar_rust.jpg | Apple cedar rust | 84% |
| plantdoc/grape_black_rot.jpg | Grape black rot | 80% |
| plantdoc/potato_late_blight.jpg | Potato late blight | 70% |

These were picked because the model gets them right. On all 236 PlantDoc test photos it is
right about 48% of the time (70% when the plant is chosen), so other field photos will do worse.

Images belong to their datasets' authors; see each source repository for its license.
