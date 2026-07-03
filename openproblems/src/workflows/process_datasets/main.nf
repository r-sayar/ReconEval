workflow auto {
  findStates(params, meta.config)
    | meta.workflow.run(
      auto: [publish: "state"]
    )
}

workflow run_wf {
  take:
  input_ch

  main:
  output_ch = input_ch
    | process_dataset.run(
      fromState: [ input: "input" ],
      toState: [
        output_train: "output_train",
        output_test: "output_test",
        output_solution: "output_solution"
      ]
    )
    | setState(["output_train", "output_test", "output_solution"])

  emit:
  output_ch
}
