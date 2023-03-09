import torch


class Batch2Loss:
    @staticmethod
    def module4(diff_main_loss, # modules
                norm_spec, decoder_inp_t, ret, K_step, batch_size, device): # variables
        '''
            training diffusion using spec as input and decoder_inp as condition.
            
            Args:
                norm_spec: (normalized) spec
                decoder_inp_t: (transposed) decoder_inp
            Returns:
                ret['diff_loss']
        '''
        t = torch.randint(0, K_step, (batch_size,), device=device).long()
        norm_spec = norm_spec.transpose(1, 2)[:, None, :, :]  # [B, 1, M, T]
        ret['diff_loss'] = diff_main_loss(norm_spec, t, cond=decoder_inp_t)
        # nonpadding = (mel2ph != 0).float()
        # ret['diff_loss'] = self.p_losses(x, t, cond, nonpadding=nonpadding)
