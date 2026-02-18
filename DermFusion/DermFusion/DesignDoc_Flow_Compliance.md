# DermaFusion iOS Design Doc Flow Compliance

Source: `DermFusion/DermFusion/DermFusion_iOS_Design_Doc.md`

## Screen Flow Compliance (Section 4.1)

- [x] Launch -> first-launch disclaimer gate
- [x] Home screen entry in Scan tab (`Views/Home/HomeView.swift`)
- [x] New Scan -> Camera/Gallery step (`Views/Scan/CameraCaptureView.swift`)
- [x] Image Review step (`Views/Scan/ImageReviewView.swift`)
- [x] Metadata Input step (`Views/Scan/MetadataInputView.swift`)
- [x] Analysis loading step (`Views/Scan/AnalysisView.swift`)
- [x] Results step with chart, GradCAM toggle, risk badge (`Views/Results/ResultsView.swift`)
- [x] History tab scaffold
- [x] Learn tab scaffold
- [x] About tab scaffold

## MVP Feature Flow Compliance (Section 2.1)

- [ ] F1 Camera capture integration (real AVFoundation)
- [ ] F1 Photo library integration (real picker)
- [ ] F1 Pinch-to-zoom + real crop review
- [x] F2 Metadata fields present (age/sex/location)
- [ ] F2 Interactive body silhouette map with front/back toggle
- [ ] F3 Real CoreML inference trigger
- [x] F3 Loading state in flow
- [x] F3 Results layout structure in place
- [ ] F3 Real probability data from model
- [ ] F4 Persistent banner on results (exact final text polish)

## Timeline Alignment (Section 7)

### Week 1
- [ ] CoreML conversion + parity + benchmarks

### Week 2
- [x] App shell/tab navigation
- [ ] Real camera and gallery pipeline
- [ ] Swift preprocessing parity validation

### Week 3
- [ ] ModelService inference
- [x] Metadata input baseline
- [ ] Body map interactive view
- [x] Results UI baseline
- [x] GradCAM toggle baseline
- [ ] Risk logic from real probabilities

### Week 4
- [ ] Persistence-backed history
- [ ] Education full content screens
- [ ] PDF export
- [x] First-launch disclaimer baseline
- [ ] Accessibility full pass

### Week 5
- [ ] Full unit + UI tests
- [ ] Performance optimization and profiling
- [ ] TestFlight/app store submission assets
