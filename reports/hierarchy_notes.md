# Experimental hierarchy reconstruction

## Original cohort
The first collection includes the Dino-Lite and Samsung A31 devices,
each under lamp-on and lamp-off conditions.

Images at corresponding concentrations were taken from the same
experimental tube preparations, according to the project owner's
description. Individual photo-to-tube and photo-to-droplet pairings
still require reconstruction and verification.

## New-tube cohort
This collection uses newly prepared tube samples, following the
same general acquisition procedure as the original cohort.

For each concentration and device, there are 12 images:
the first six are lamp-on and the next six are lamp-off.

The exact number of physical tubes and droplets represented by these
images has not yet been confirmed. Do not infer tube or droplet IDs
solely from image order.

## November 4 validation cohort
This is a separate collection from the third prepared sample.
There are four Dino-Lite images per recorded concentration:
the first two are lamp-on and the next two are lamp-off.

The number of droplets and the exact photo-to-droplet correspondence
are currently unknown. Keep this cohort separate from development
and model selection.

## Identity policy
Do not silently invent tube IDs, droplet IDs or view IDs.
Use filenames, timestamps, sequence numbers, visual similarity and
available experimental records as evidence.
Record unresolved identities explicitly. Where tube identity cannot
be recovered, follow the conservative fallback in the roadmap's Phase 3.

## Known unresolved observations
- Dino-Lite lamp-off sequence A056 appears twice at 0.40% on
  September 11, 2024; investigate whether these are distinct
  observations and whether they belong to different droplets/tubes.
- The original Dino-Lite lamp-off condition has four photos at
  0.20% and four at 0.40%.
- The original Dino-Lite lamp-on condition has five photos at 0.45%.
- The original Samsung A31 lamp-off condition has eight photos at
  0.20%; the lamp-on condition has nine.
- Forty-seven original Samsung A31 lamp-off filenames do not contain
  timestamps, and their EXIF data do not provide timestamps.
