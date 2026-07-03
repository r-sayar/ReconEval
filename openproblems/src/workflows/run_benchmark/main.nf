workflow auto {
  findStates(params, meta.config)
    | meta.workflow.run(
      auto: [publish: "state"]
    )
}

methods = [
  ground_truth,
  pca_reconstruction,
  autoencoder,
  scvi,
]

metrics = [
  statistical,
]

workflow run_wf {
  take:
  input_ch

  main:

  dataset_ch = input_ch
    | map { id, state ->
      [id, state + ["_meta": [join_id: id]]]
    }
    | extract_uns_metadata.run(
      fromState: [input: "input_solution"],
      toState: { id, output, state ->
        state + [dataset_uns: readYaml(output.output).uns]
      }
    )

  score_ch = dataset_ch
    | runEach(
      components: methods,
      filter: { id, state, comp ->
        def norm = state.dataset_uns.normalization_id
        def pref = comp.config.info.preferred_normalization
        def norm_check = norm == pref
        def method_check = !state.method_ids || state.method_ids.contains(comp.config.name)
        method_check && norm_check
      },
      id: { id, state, comp ->
        id + "." + comp.config.name
      },
      fromState: { id, state, comp ->
        def args = [
          input_train: state.input_train,
          input_test: state.input_test,
        ]
        if (comp.config.info.type == "control_method") {
          args.input_solution = state.input_solution
        }
        args
      },
      toState: { id, output, state, comp ->
        state + [
          method_id: comp.config.name,
          method_output: output.output,
        ]
      }
    )
    | runEach(
      components: metrics,
      id: { id, state, comp ->
        id + "." + comp.config.name
      },
      fromState: [
        input_solution: "input_solution",
        input_prediction: "method_output",
      ],
      toState: { id, output, state, comp ->
        state + [
          metric_id: comp.config.name,
          metric_output: output.output,
        ]
      }
    )
    | extract_uns_metadata.run(
      key: "extract_scores",
      fromState: [input: "metric_output"],
      toState: { id, output, state ->
        state + [score_uns: readYaml(output.output).uns]
      }
    )
    | joinStates { ids, states ->
      def score_uns = states.collect { it.score_uns }
      def score_uns_yaml_blob = toYamlBlob(score_uns)
      def score_uns_file = tempFile("score_uns.yaml")
      score_uns_file.write(score_uns_yaml_blob)
      ["output", [output_scores: score_uns_file]]
    }

  meta_ch = dataset_ch
    | joinStates { ids, states ->
      def dataset_uns = states.collect { state ->
        def uns = state.dataset_uns.clone()
        uns.remove("normalization_id")
        uns
      }
      def dataset_uns_file = tempFile("dataset_uns.yaml")
      dataset_uns_file.write(toYamlBlob(dataset_uns))

      def method_configs_file = tempFile("method_configs.yaml")
      method_configs_file.write(toYamlBlob(methods.collect { it.config }))

      def metric_configs_file = tempFile("metric_configs.yaml")
      metric_configs_file.write(toYamlBlob(metrics.collect { it.config }))

      def viash_file = meta.resources_dir.resolve("_viash.yaml")
      def new_state = [
        output_dataset_info: dataset_uns_file,
        output_method_configs: method_configs_file,
        output_metric_configs: metric_configs_file,
        output_task_info: viash_file,
        _meta: states[0]._meta,
      ]
      ["output", new_state]
    }

  output_ch = score_ch
    | mix(meta_ch)
    | joinStates { ids, states ->
      [ids[0], states.inject([:]) { acc, m -> acc + m }]
    }

  emit:
  output_ch
}
