
### Metrics
- [ ] loss logging
- [ ] agent performance (eval against real env)
- [ ] sample images

### Fix Model
- [ ] fix terminal loss grads
- [ ] implement prob dist sampling inside model layers

### Support various Gym Envs
- [ ] sample on continuous action spaces (not only discrete)
- [ ] support inputs other that images (make encoder exchangeable)

### Backup Model
- [ ] add model snapshots during training
- [ ] union agent + dream encoder as inference agent acting on real data

### Optimize Performance
- [ ] vectorize real env for collecting trajectories
- [ ] track down memory leak

### Add Docs
- [ ] explain dreamer architecture (how the model works)
- [ ] write idiot-proof installation for windows / linux
- [ ] add fancy github badges, etc. ^^
