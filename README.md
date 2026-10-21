# voxels
redo of cpu vox project.
my lack of uploading to git finally bit me in the ass and i lost it all :(.
starting from scratch and trying to hit actual milestones

its a voxel ray tracer that runs completely on the cpu, no gpu shaders, the gpu just stretches the finished image onto the screen.
the world is 256m x 256m x 256m made of 0.125m voxels, so 2048³ voxels (about 8.6 billion cells) packed into about 80mb.


# current planned milestones
- render one small volume of periodically changing materials   DONE
- reduce voxel size   DONE
- implement a basic contree   DONE (64-tree now)
- semi-realistic terrain test   DONE
- realistic terrain test   DONE-ish (hills, rounded mountains, lakes, rivers, trees, snow up top)
- collision + walking around   DONE
- dev menu + debug overlays   DONE
- breaking + placing voxels   DONE (1 voxel up to a 1m ball)
- solid 30fps at 720p near the ground   not yet, getting there
- bigger world (1km+)   later


# how it works (the short version)

### the tree
- 64-tree (contree), every branch is a 4x4x4 grid with a 64 bit mask for which cells have children
- children are stored packed, so finding a child is just `child_pointer + popcount(mask below the cell)`
- cells that are completely solid or completely water get their own bits so they never need a child
- leaves are 8x8x8 voxels (one cache line of occupancy bits), materials sit in a separate 4 bit per voxel brick that only gets read once a ray actually hits something
- every leaf also knows which of its 2x2x2 blocks have anything in them, so rays hop over empty blocks
- a skip map (one byte per meter) stores how far away the nearest solid meter is, so rays can jump whole empty cubes instead of stepping cell by cell. theres a second one that ignores water for rays looking through it

### tracing
- rays go out in packets of 4 using 4 wide simd (arm neon for now, the x86 side is still just stubs so it only runs on apple silicon / arm)
- the screen is cut into 16x16 tiles walked in morton order and spread across every core with rayon, tiles never overlap so no locks
- traversal keeps an ancestor stack, the highest bit that changed on a step says how far back up the tree to go
- camera rays get a head start: every 8x8 pixel block marches its center ray through the skip map, and as long as the whole fan of its rays fits inside the empty cubes it can jump ahead. all 16 packets of the block start from there, saves about a fifth of the frame near the ground
- empty 2x2x2 groups of cells get stepped over as one bigger cell, and rays in the same packet that land in the same cell just share the answer instead of walking the tree again. both of those came from reading [dubiousconst282's 64-tree write up](https://dubiousconst282.github.io/2024/10/03/voxel-ray-tracing/)
- lod ladder: full res up to 24m, then rays hit 2x2x2 voxel blocks as a whole, 4x4x4 from 48m and a whole 8x8x8 meter from 96m. a lumped block looks like the material of its highest voxel. shadow rays lump any cell no wider than them

### lighting (voxel lighting)
- light gets stored per voxel face in the world, not per pixel. every pixel looking at a face just reads its light, so a whole voxel side is one flat color and up close it looks properly voxel-y
- the faces live in one big hash table (2 million faces, 64mb), the first lane that sees a face each frame decides if it needs shading, nobody else touches it. no locks, just atomics
- the light patches follow the same lod ladder: every face has its own light up to 24m, then 2x2 / 4x4 / 8x8 voxel patches at 24 / 48 / 96m
- a face sample is a sun shadow ray from a random spot on the face plus one bounce. if the bounce hits a face thats already in the cache it just borrows its light, so every bounce after the first is free
- new faces get 4 samples right away, young ones get shaded every frame, settled ones only every so often (the frame budget decides how often)
- sun is a square light you can move from the dev menu, low sun gets dimmer and redder. moving it forgets every face (costs nothing, it just bumps a generation number)
- water does a reflection ray and a refraction ray mixed by fresnel, you can see the bottom up to about 6m, deeper is just water color. going under water fogs everything out
- reflections go 32m, past that they show sky. they show the cached light of whatever they hit
- water closer than 32m traces a reflection for every pixel, further out only every 2nd / 4th pixel does and the rest blur together the ones around them, blurrier the further out

### moving without it looking like garbage
- since the light lives in the world, moving the camera doesnt throw any of it away. moving frames look just as clean as still ones, no history, no reprojection, no smearing
- every pixel still gets its own camera ray so stuff stays sharp
- standing still the rays get jittered and every pixel keeps a running mean of its color, thats the anti aliasing
- breaking / placing stuff makes the faces around it young again so the light there catches up in a few frames

### terrain
- noise height field with thermal erosion so mountains turn into rounded hills instead of spikes
- lakes are found with a priority flood and filled per voxel column so they stay flat and never drain into weird staircases (that was the melted dirt bug lol)
- rivers come from flow accumulation and only exist where the ground is gentle enough to hold them
- trees are just shapes until a leaf actually needs them

### editing
- the tree is packed tight for tracing (siblings sit side by side), so giving a branch a new child copies its whole family to the end of the pool with the new one slotted in. a family thats already at the end just grows in place
- solid / water cells get split open into children just as full as they were, all the way down to a leaf
- the skip map only ever shrinks around new stuff, breaking things leaves it alone since too small is still correct, rays just take a couple extra steps
- holes dug next to or under water flood (from above or the side, never upwards)
- an edit takes well under a millisecond even at the biggest brush


# controls
- wasd move, mouse look, shift sprint
- space up / c down while flying, flying speeds up the longer you hold a direction
- double tap space to switch between flying and walking (gravity, jumping, swimming)
- left click breaks, right click places, hold either to keep going
- scroll grows / shrinks the brush, 1 voxel up to an 8 voxel (1m) ball, the ground under it lights up so you can see how big it is
- arrow keys cycle what's in hand, middle click picks up whatever you're looking at
- 1 debug overlay, 2 performance overlay (cpu, ram, frame graph), 3 dev menu (w/s pick, a/d change, 3 or esc to close)
- 0 fullscreen (render resolution stays the same, the gpu just stretches it)


# running it
```
cargo run --release
```
never run it in debug, it runs at like 0.5 fps

the first run generates the world (~2s) and caches it in `cache/world_42.bin` (~130mb), after that it asks if you want to reuse it, loading takes like 20ms. the cache knows which version of the generator made it, so changing the terrain code just makes it build a new one
```
cargo run --release -- --reuse       # use the cache, don't ask
cargo run --release -- --new-world   # always generate a fresh one
```

tests (traversal gets checked against a dumb exact reference walk on 13k+ rays, plus lakes, collision and walking):
```
cargo test --release
```


# performance (m2 macbook air, 720p, moving)
- looking down from high up: ~21ms
- hills near the ground: ~35ms
- over a lake: ~60ms (near water still traces a reflection + refraction for every pixel every frame)
- shading the faces is basically free now (a couple % of the frame), its almost all camera rays and water rays

camera rays are the next thing to fix, thats whats keeping it from 30fps near the ground


Demo of small volume (old version):

https://youtu.be/yy-teP9txk4
