window.BENCHMARK_DATA = {
  "lastUpdate": 1776566709238,
  "repoUrl": "https://github.com/harmoniqs/CuQuantum.jl",
  "entries": {
    "CuQuantum.jl GPU benchmarks (g4dn.xlarge)": [
      {
        "commit": {
          "author": {
            "name": "Jack Champagne",
            "username": "jack-champagne",
            "email": "jack@harmoniqs.co"
          },
          "committer": {
            "name": "Jack Champagne",
            "username": "jack-champagne",
            "email": "jack@harmoniqs.co"
          },
          "id": "c3865d966344011b240b47ea4fb729490246967b",
          "message": "ci(bench): allow workflow_dispatch on main to publish to /bench\n\nThe prior gate required a push event, but the workflow only has\nworkflow_dispatch as a trigger — so the gate never evaluated true\nand no datapoints could ever land on gh-pages. Simplify to a ref-only\ncheck so manual dispatches on main publish.",
          "timestamp": "2026-04-19T01:06:02Z",
          "url": "https://github.com/harmoniqs/CuQuantum.jl/commit/c3865d966344011b240b47ea4fb729490246967b"
        },
        "date": 1776566708885,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "cuDensityMat GPU L[ρ] M=2 D=9",
            "value": 0.21109,
            "unit": "ms"
          },
          {
            "name": "QT.jl GPU cuSPARSE L[ρ] M=2 D=9",
            "value": 0.032517,
            "unit": "ms"
          },
          {
            "name": "CPU dense SpMV L[ρ] M=2 D=9",
            "value": 0.0025989999999999997,
            "unit": "ms"
          },
          {
            "name": "QT.jl CPU sparse L[ρ] M=2 D=9",
            "value": 0.000705,
            "unit": "ms"
          },
          {
            "name": "cuDensityMat GPU L[ρ] M=4 D=81",
            "value": 3.3150135,
            "unit": "ms"
          },
          {
            "name": "QT.jl GPU cuSPARSE L[ρ] M=4 D=81",
            "value": 0.050157499999999994,
            "unit": "ms"
          },
          {
            "name": "CPU dense SpMV L[ρ] M=4 D=81",
            "value": 31.1145585,
            "unit": "ms"
          },
          {
            "name": "QT.jl CPU sparse L[ρ] M=4 D=81",
            "value": 0.20551550000000002,
            "unit": "ms"
          },
          {
            "name": "cuDensityMat GPU L[ρ] M=6 D=729",
            "value": 148.7875175,
            "unit": "ms"
          },
          {
            "name": "QT.jl GPU cuSPARSE L[ρ] M=6 D=729",
            "value": 2.408922,
            "unit": "ms"
          },
          {
            "name": "QT.jl CPU sparse L[ρ] M=6 D=729",
            "value": 41.9881805,
            "unit": "ms"
          },
          {
            "name": "cuDensityMat GPU L[ρ] M=8 D=6561",
            "value": 17067.896175,
            "unit": "ms"
          }
        ]
      }
    ]
  }
}