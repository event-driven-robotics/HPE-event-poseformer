# 3D HPE using event-PoseFormer

The code for the paper
> Melica Omer Ali et. al. "EventPoseFormer: Event-based real-time 3D human pose estimation"


```
@inproceedings{ali2026,
  title={EventPoseFormer: Event-based real-time 3D human pose estimation},
  author={},
  booktitle={venue},
  pages={},
  year={2026},
  organization={IEEE}
}
```

## How to run the code

**Offline Execution**
```bash
python evpf-offline.py --event_input_path "inputs/cam2_S1_Directions/ch0dvs" --output_dir "outputs" --skip 0-10 --save_images --save_demo
```

**Offline Evaluation**
```bash
python 3d_evaluate.py --pred_path outputs/cam2_S1_Directions/predictions_3d.npz --subject S1 --activity Directions --frame 1000 --plot
```

**Online Execution**

- Install moveEnet and atis-gen-3 docker images from here: [moveEnet](https://github.com/event-driven-robotics/hpe-core/blob/main/example/movenet/Dockerfile) and [atis-gen-3](https://hub.docker.com/r/eventdrivenrobotics/atis-gen3)
- Run the following command to start the moveEnet docker container:
```bash
docker run -it --privileged --network host -v /dev/bus/usb:/dev/bus/usb -v /tmp/.X11-unix:/tmp/.X11-unix  -v yourpathtohpecore:/mnt/projects/hpe-core -e DISPLAY=unix$DISPLAY --name moveenet moveenet:latest
```
- Inside the moveEnet docker container, run the following commands in separate terminals:

```bash
yarpserver
```
```bash
eros-framer
```

*Note*: Run the following command to enter the moveEnet docker container from another terminal:
```bash
docker exec -it moveenet bash
```

- In another terminal, run the following command to start the atis-gen-3 docker container:
```bash
lsusb
```
Use the output of the above command to find the USB device ID of your ATIS camera. Then run:
```bash
docker run -it --network host --device /dev/bus/usb/[ID]:/dev/bus/usb/[ID] eventdrivenrobotics/atis-gen3
```

- Inside the atis-gen-3 docker container, run the following commands:
```bash
yarp detect --write
```
```bash
atis-bridge-sdk --s 60 --filter 0.01
```
- Run each of the following commands in separate terminals inside the moveEnet docker container:
```bash
python3 evpf-online.py --checkpoint_path /usr/local/hpe-core/example/movenet/models/e97_valacc0.81209.pth
```
```bash
python3 movenet_viewer.py --name /movenetViewer
```
```bash
yarpview --name /viewer
```
```bash
python3 evpf-viewer1.py --name /evpfViewer
```
```bash
yarp connect /atis3/AE:o /eroser/AE:i
yarp connect /eroser/img:o /movenet/img:i
yarp connect /eroser/img:o /movenetViewer/img:i
yarp connect /movenet/sklt:o /movenetViewer/sklt:i
yarp connect /movenetViewer/overlay:o /viewer
yarp connect /movenet/sklt3d:o /evpfViewer/sklt3d:i
```