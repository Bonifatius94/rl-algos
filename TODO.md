
### Metrics
- [x] loss logging
- [ ] agent performance (eval against real env)
- [x] sample images
- [ ] log the representation states

### Fix Model
- [x] fix terminal loss grads
- [ ] implement prob dist sampling inside model layers

### Support various Gym Envs
- [ ] sample on continuous action spaces (not only discrete)
- [ ] support inputs other than images (make encoder/decoder exchangeable)

### Backup Model
- [x] add model snapshots during training
- [ ] ~~union agent + dream encoder as inference agent acting on real data~~

### Optimize Performance
- [ ] vectorize real env for collecting trajectories
- [ ] track down memory leak

### Add Docs
- [ ] explain dreamer architecture (how the model works)
- [x] write idiot-proof installation for windows / linux
- [ ] add fancy github badges, etc. ^^
