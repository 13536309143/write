import copy
import tempfile
import unittest
from pathlib import Path
import torch
import yaml
from handwriting.model import HandwritingNet
from handwriting.runtime import (load_config, precision_for, training_signature,
                                  restore_scaler, optimizer_update)

ROOT = Path(__file__).resolve().parents[1]


class OptimizationTests(unittest.TestCase):
    def test_amp_overflow_skips_parameters_then_recovers(self):
        # CPU GradScaler runs the real scaling/overflow mechanism on this Mac;
        # CUDA kernel coverage is a separate hardware-dependent test below.
        model = torch.nn.Linear(2,2)
        optimizer = torch.optim.AdamW(model.parameters(),lr=0.01)
        scaler = torch.amp.GradScaler('cpu',init_scale=32)
        original = [p.detach().clone() for p in model.parameters()]
        scaler.scale(model(torch.ones(2,2)).square().mean()).backward()
        model.weight.grad[0,0] = float('inf')
        self.assertFalse(optimizer_update(model,optimizer,scaler,1.0))
        self.assertEqual(scaler.get_scale(),16)
        self.assertFalse(optimizer.state)
        for a,b in zip(original,model.parameters()):torch.testing.assert_close(a,b)
        scaler.scale(model(torch.ones(2,2)).square().mean()).backward()
        self.assertTrue(optimizer_update(model,optimizer,scaler,1.0))
        self.assertTrue(optimizer.state)
        self.assertTrue(any(not torch.equal(a,b) for a,b in zip(original,model.parameters())))

    def test_fp32_invalid_gradient_is_fatal(self):
        model = torch.nn.Linear(2,2)
        optimizer = torch.optim.AdamW(model.parameters())
        scaler = torch.amp.GradScaler('cpu',enabled=False)
        model(torch.ones(1,2)).sum().backward()
        model.weight.grad[0,0] = float('nan')
        original = model.weight.detach().clone()
        with self.assertRaises(RuntimeError):optimizer_update(model,optimizer,scaler,1.0)
        torch.testing.assert_close(original,model.weight)

    def test_empty_and_existing_scaler_restore(self):
        scaler = torch.amp.GradScaler('cpu',init_scale=32)
        restore_scaler(scaler,{})
        self.assertEqual(scaler.get_scale(),32)
        source = torch.amp.GradScaler('cpu',init_scale=128)
        restore_scaler(scaler,source.state_dict())
        self.assertEqual(scaler.get_scale(),128)
        disabled = torch.amp.GradScaler('cpu',enabled=False)
        restore_scaler(disabled,source.state_dict())
        self.assertEqual(disabled.state_dict(),{})

    def test_runtime_changes_allow_resume_but_training_changes_do_not(self):
        mac = load_config(ROOT/'configs/mac.yaml')
        windows = load_config(ROOT/'configs/windows_cuda.yaml')
        self.assertEqual(training_signature(mac),training_signature(windows))
        changed = copy.deepcopy(windows)
        changed['batch_size'] *= 2
        self.assertNotEqual(training_signature(mac),training_signature(changed))
        changed = copy.deepcopy(windows)
        changed['epochs'] += 1
        self.assertNotEqual(training_signature(mac),training_signature(changed))
        self.assertEqual(precision_for(mac,torch.device('cpu')),'fp32')
        self.assertEqual(precision_for(windows,torch.device('cuda')),'fp16')
        with self.assertRaises(ValueError):precision_for(windows,torch.device('cpu'))

    def test_utf8_bom_config_and_relative_unicode_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)/'configs';folder.mkdir()
            path = folder/'windows.yaml'
            path.write_text(yaml.safe_dump({'data_dir':'数据/processed','run_dir':'结果/训练'},allow_unicode=True),
                            encoding='utf-8-sig')
            config = load_config(path)
            self.assertEqual(Path(config['data_dir']),Path(directory).resolve()/'数据/processed')
            self.assertEqual(Path(config['run_dir']),Path(directory).resolve()/'结果/训练')

    @unittest.skipUnless(torch.cuda.is_available(),'Requires an NVIDIA CUDA GPU')
    def test_cuda_network_fp16_updates(self):
        model = HandwritingNet(7247).cuda()
        optimizer = torch.optim.AdamW(model.parameters(),lr=0.0001)
        scaler = torch.amp.GradScaler('cuda')
        images = torch.randn(2,1,128,128,device='cuda')
        target = torch.tensor([0,62],device='cuda')
        applied = 0
        for _ in range(10):
            with torch.autocast('cuda',dtype=torch.float16):
                loss = torch.nn.functional.cross_entropy(model(images),target)
            self.assertTrue(bool(torch.isfinite(loss)))
            scaler.scale(loss).backward()
            applied += optimizer_update(model,optimizer,scaler,1.0)
            if applied >= 2:break
        self.assertGreaterEqual(applied,2)


if __name__ == '__main__':unittest.main()
