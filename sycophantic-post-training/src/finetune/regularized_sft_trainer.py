import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from trl import SFTTrainer


class RegularizedSFTTrainer(SFTTrainer):
    def __init__(
        self,
        *args,
        reference_model,
        regularization_dataset: Dataset,
        regularization_alpha: float = 0.1,
        **kwargs,
    ):
        """
        Set reference_model to None if model is a peft model.
        """
        super().__init__(*args, **kwargs)

        self.regularization_dataset = (
            regularization_dataset  # the only field here will be 'text'
        )
        print(f"Regularization dataset size: {len(self.regularization_dataset)}")
        print(
            f"Regularization dataset columns: {self.regularization_dataset.column_names}"
        )
        print("About to map regularization dataset")
        self.regularization_dataset = self.regularization_dataset.map(
            self.prepare_regularization_inputs,
            batched=True,
            remove_columns=["text"],
            num_proc=1,
        )
        print(
            f"Regularization dataset after mapping: {self.regularization_dataset[:3]}"
        )
        print(
            f"Regularization dataset columns after mapping: {self.regularization_dataset.column_names}"
        )
        self.regularization_dataloader = DataLoader(
            self.regularization_dataset,
            batch_size=self.args.per_device_train_batch_size,
            shuffle=True,
            collate_fn=self.data_collator,
        )

        self.regularization_iterator = iter(self.regularization_dataloader)
        self.regularization_alpha = regularization_alpha
        self.reference_model = reference_model

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        inputs["labels"] = inputs["labels"].to(model.device)
        inputs["input_ids"] = inputs["input_ids"].to(model.device)
        if inputs["labels"].shape != inputs["input_ids"].shape:
            raise ValueError(
                "Labels and input IDs must have the same shape. otherwise the regularization loss approach is wrong"
            )
        for i in range(inputs["labels"].shape[1]):
            for j in range(inputs["labels"].shape[0]):
                if inputs["labels"][j, i] != -100:
                    assert inputs["input_ids"][j, i] == inputs["labels"][j, i], (
                        "Labels have been pre-shifted by one"
                    )

        if return_outputs:
            print("About to call super().compute_loss(return_outputs=True)")
            loss, outputs = super().compute_loss(
                model, inputs, return_outputs=True, **kwargs
            )
        else:
            print("About to call super().compute_loss(return_outputs=False)")
            loss = super().compute_loss(model, inputs, return_outputs=False, **kwargs)
            outputs = None

        print("About to call compute_regularization_loss")
        print("What device is the model on? ", model.device)
        reg_loss = self.compute_regularization_loss(model)
        total_loss = loss + self.regularization_alpha * reg_loss
        print(f"Loss: {loss}")
        print(f"Regularization loss: {reg_loss}")
        print(f"Total loss: {total_loss}")
        return (total_loss, outputs) if return_outputs else total_loss

    def get_batch_logprobs(self, model, inputs, no_grad=False):
        """
        Get the log probabilities of the model's outputs.
        """
        inputs = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
        }
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        if no_grad:
            with torch.no_grad():
                logits = model(**inputs).logits[:, :-1, :]
        else:
            logits = model(**inputs).logits[:, :-1, :]
        logprobs = torch.log_softmax(logits, dim=-1)

        return logprobs

    def prepare_regularization_inputs(self, inputs: dict[str, list]):
        """
        Prepare the regularization inputs for the model.
        """
        print(f"Fields found in regularization inputs: {inputs.keys()}")
        processed_inputs = self.processing_class(inputs["text"])
        return processed_inputs

    def compute_regularization_loss(self, model):
        """
        We will compute regularization between model and reference model on these completions.
        I expect the regularization dataset to be in messages format, e.g. have a messages key with a list of messages.
        """
        try:
            reg_inputs = next(self.regularization_iterator)
        except StopIteration:
            self.regularization_iterator = iter(self.regularization_dataloader)
            reg_inputs = next(self.regularization_iterator)

        if hasattr(model, "disable_adapter"):
            with model.disable_adapter():
                ref_logprobs = self.get_batch_logprobs(model, reg_inputs, no_grad=True)
        else:
            assert self.reference_model is not None, (
                "Reference model must be provided if model is not a peft model"
            )
            ref_logprobs = self.get_batch_logprobs(
                self.reference_model, reg_inputs, no_grad=True
            )
        model_logprobs = self.get_batch_logprobs(model, reg_inputs, no_grad=False)

        response_mask = (reg_inputs["labels"][:, 1:] != -100).float().to(model.device)

        kl_div = torch.exp(model_logprobs) * (
            model_logprobs - ref_logprobs
        )  # should be shape batch size, seq len, vocab size
        print("KL div device: ", kl_div.device)
        print("Response mask device: ", response_mask.device)
        kl_div = kl_div.sum(dim=-1) * response_mask
        kl_div = kl_div.sum() / response_mask.sum()
        return kl_div
