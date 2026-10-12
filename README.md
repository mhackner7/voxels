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
- solid 30fps at 720p near the ground   not yet, getting there
- bigger world (1km+)   later


## the tree
- 64-tree (contree), every branch is a 4x4x4 grid with a 64 bit mask for which cells have children
- children are stored packed, so finding a child is just `child_pointer + popcount(mask below the cell)`
- cells that are completely solid or completely water get their own bits so they never need a child
- leaves are 8x8x8 voxels (one cache line of occupancy bits), materials sit in a separate 4 bit per voxel brick that only gets read once a ray actually hits something
- every leaf also knows which of its 2x2x2 blocks have anything in them, so rays hop over empty blocks
- a skip map (one byte per meter) stores how far away the nearest solid meter is, so rays can jump whole empty cubes instead of stepping cell by cell. theres a second one that ignores water for rays looking through it

## tracing
- rays go out in packets of 4 using 4 wide simd (arm neon for now, the x86 side is still just stubs so it only runs on apple silicon / arm)
- the screen is cut into 16x16 tiles walked in morton order and spread across every core with rayon, tiles never overlap so no locks
- traversal keeps an ancestor stack, the highest bit that changed on a step says how far back up the tree to go
- empty 2x2x2 groups of cells get stepped over as one bigger cell, and rays in the same packet that land in the same cell just share the answer instead of walking the tree again. both of those came from reading [dubiousconst282's 64-tree write up](https://dubiousconst282.github.io/2024/10/03/voxel-ray-tracing/)
- lod: once a ray is wider than a 2x2x2 block it just hits the block as a whole, shadow rays do it with any cell

## lighting
- sun is a square light you can move from the dev menu, low sun gets dimmer and redder
- every pixel gets a sun shadow ray plus one bounce (with its own shadow ray), it accumulates up to 512 samples while you stand still
- water does a reflection ray and a refraction ray mixed by fresnel, you can see the bottom up to about 6m, deeper is just water color. going under water fogs everything out

## moving without it looking like garbage
- still frames keep piling up samples, but when you move the old image gets reprojected instead of thrown away
- every pixel still gets its own camera ray so stuff stays sharp, but the lighting gets reused from last frame if that spot was on screen already. only new spots and a rotating handful of others get re-lit
- history only holds the light, not the texture, so textures and voxel edges dont smear
- history only gets reused from the same voxel face at the same depth, so it never leaks across edges
- the frame budget (60 / 30 / off in the dev menu) decides how much gets re-lit each frame

## terrain
- noise height field with thermal erosion so mountains turn into rounded hills instead of spikes
- lakes are found with a priority flood and filled per voxel column so they stay flat and never drain into weird staircases (that was the melted dirt bug lol)
- rivers come from flow accumulation and only exist where the ground is gentle enough to hold them
- trees are just shapes until a leaf actually needs them


## controls
- wasd move, mouse look, shift sprint
- space up / c down while flying, flying speeds up the longer you hold a direction
- double tap space to switch between flying and walking (gravity, jumping, swimming)
- 1 debug overlay, 2 performance overlay (cpu, ram, frame graph), 3 dev menu (w/s pick, a/d change, 3 or esc to close)
- 0 fullscreen (render resolution stays the same, the gpu just stretches it)


### performance (m2 macbook air, 720p, moving)
- looking down from high up: ~24ms
- hills near the ground: ~41ms
- over a lake: ~49ms
- standing still always fits the budget, it just splits the image into slices and sharpens over a few frames

near the ground the camera rays alone are most of the frame now, so thats the next thing to fix


Demo of small volume (old version):

https://youtu.be/yy-teP9txk4
