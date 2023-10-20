import torch


class BaseModelOuput:

    def __init__(self, **kwarg):
        self.logits = kwarg.get('logits')
        self.embeds = kwarg.get('embeds')
        self.decoder_hidden = kwarg.get('decoder_hidden')

    @torch.no_grad()
    def log_value(self, writer, step, val, name):
        val = val.cpu()
        q_pos = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])

        q_val = torch.quantile(val, q_pos, dim=-1)
        # torch.quantile(val[0], q)
        writer.add_scalar(f'{name}/min', q_val[0].item(), step)
        writer.add_scalar(f'{name}/q1', q_val[1].item(), step)
        writer.add_scalar(f'{name}/q2', q_val[2].item(), step)
        writer.add_scalar(f'{name}/q3', q_val[3].item(), step)
        writer.add_scalar(f'{name}/max', q_val[4].item(), step)

        writer.add_scalar(f'{name}/mean', val.mean().item(), step)
        writer.add_scalar(f'{name}/std', val.std().item(), step)
        writer.add_scalar(f'{name}/sum', val.sum().item(), step)


class SegmentOutput(BaseModelOuput):

    def __init__(self, **kwarg):
        super().__init__(**kwarg)

    def log_model_value(self, writer, step):
        self.log_value(writer, step, self.decoder_hidden[0].sum(dim=-1), 'decoder_hidden')
