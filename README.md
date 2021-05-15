# Mandelbrot Visualizer

## Example usage:
```
python find.py 1500 -z 1.005 -p 6 -r
python render.py 1500 result.mp4 -z 1.005 -l 50 -ci -p 4 -gs 48 -cm PuBu
```

## find.py --help
```
usage: find.py [-h] [-bw [BW]] [-bh [BH]] [-gw [GW]] [-gh [GH]] [-r] [-z [Z]] [-p [P]] [-l [L]] steps

Mandelbrot Set conntrasting points finder.

positional arguments:
  steps       Number of steps

optional arguments:
  -h, --help  show this help message and exit
  -bw [BW]    Block width
  -bh [BH]    Block height
  -gw [GW]    Grid width
  -gh [GH]    Grid height
  -r          Render results
  -z [Z]      Zoom speed
  -p [P]      Precision
  -l [L]      Iterations limit
```

## render.py --help
```
usage: render.py [-h] [-bs [BS]] [-gs [GS]] [-l [L]] [-x [X]] [-y [Y]] [-fps [FPS]] [-z [Z]] [-cm [CM]] [-i] [-nf] [-ci] [-lp] [-p [P]] frames output

Mandelbrot Set zoom animation generator.

positional arguments:
  frames      Number of rendered frames
  output      Output file (mp4)

optional arguments:
  -h, --help  show this help message and exit
  -bs [BS]    Block size
  -gs [GS]    Grid size
  -l [L]      Iterations limit
  -x [X]      X offset
  -y [Y]      Y offset
  -fps [FPS]  Y offset
  -z [Z]      Zoom speed
  -cm [CM]    Matplotlib colormap
  -i          Invert colormap
  -nf         Don't save frames
  -ci         Contrast improvement
  -lp         Go to last position
  -p [P]      Precision
```
