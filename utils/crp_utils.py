import copy
from collections import Callable
from typing import List, Union, Dict

import torch
from crp.attribution import CondAttribution, attrResult
from crp.concepts import Concept, ChannelConcept
from crp.hooks import MaskHook
from crp.receptive_field import ReceptiveField, AllFlatComposite
from zennit.composites import NameMapComposite, SpecialFirstLayerMapComposite, LAYER_MAP_BASE
from zennit.core import Composite
from zennit.rules import Gamma, Epsilon, Flat
from zennit.types import Convolution, Linear


class ReceptiveFieldCRP(ReceptiveField):
    def analyze_layer(self, concept: Concept, layer_name: str, c_indices, canonizer=None, batch_size=16, verbose=True):

        composite = AllFlatComposite(canonizer)
        conditions = [{layer_name: [index]} for index in c_indices]
        idc = 0
        def mask_rf(batch_id, neuron_ids, layer_name=None):
            def mask_fct(grad):
                grad_shape = grad.shape
                grad = grad.view(*grad_shape[:2], -1)

                mask = torch.zeros_like(grad[batch_id])
                mask[:, neuron_ids] = 1  # have to apply to all because some or set to zero??

                grad[batch_id] = grad[batch_id] * mask
                return grad.view(grad_shape)

            return mask_fct

        for attr in self.attribution.generate(
                torch.ones_like(self.single_sample).requires_grad_(), conditions, composite, [], mask_rf,
                layer_name, 1, batch_size, None, verbose):

            heat = self.norm_rf(attr.heatmap, layer_name)

            try:
                rf_array[idc:(idc+len(heat))] = heat
            except UnboundLocalError:
                rf_array = torch.zeros((len(c_indices), *heat.shape[1:]), dtype=torch.uint8)
                rf_array[idc:(idc+len(heat))] = heat

            idc += len(heat)
        return rf_array


class CondAttributionDiff(CondAttribution):
    def __init__(self, model: torch.nn.Module):
        super().__init__(model)
        self.model_copy = copy.deepcopy(model)
        self.take_prediction = 0

    def __call__(
            self, data: torch.tensor, conditions: List[Dict[str, List]],
            composite: Composite = None, record_layer: List[str] = [],
            mask_map: Union[Callable, Dict[str, Callable]] = ChannelConcept.mask, start_layer: str = None,
            init_rel=None,
            on_device: str = None) -> attrResult:
        # with condition on layer (conditional relevance and relevance of other paths)

        data, conditions = self.broadcast(data, conditions)

        self.check_arguments(data, conditions, start_layer)

        handles, layer_out = self._append_recording_layer_hooks(
            record_layer, start_layer)

        hook_map, y_targets = {}, []
        for i, cond in enumerate(conditions):
            for l_name, indices in cond.items():
                if l_name == self.MODEL_OUTPUT_NAME:
                    y_targets.append(indices)
                else:
                    if l_name not in hook_map:
                        hook_map[l_name] = MaskHook([])
                    self.register_mask_fn(hook_map[l_name], mask_map, i, indices, l_name)

        name_map = [([name], hook) for name, hook in hook_map.items()]
        mask_composite = NameMapComposite(name_map)

        if composite is None:
            composite = Composite()

        with mask_composite.context(self.model), composite.context(self.model) as modified:

            if start_layer:
                _ = modified(data)
                pred = layer_out[start_layer].clamp(min=0)
                self.backward_initialization(pred, None, init_rel, start_layer)

            else:
                pred = modified(data)
                self.backward_initialization(pred, y_targets, init_rel, self.MODEL_OUTPUT_NAME)

            attribution = self.attribution_modifier(data)
            activations, relevances = {}, {}
            if len(layer_out) > 0:
                activations, relevances = self._collect_hook_activation_relevance(
                    layer_out)
            [h.remove() for h in handles]

        attr_result = attrResult(attribution, activations, relevances, pred)

        # attr_result = CondAttribution.__call__(self, copy.copy(data), conditions, composite, record_layer, mask_map,
        #                                        start_layer, init_rel, on_device)
        multiple_conditions = [condition for condition in conditions if (len(condition) > 1)]
        if not multiple_conditions:
            return attr_result

        # for conditonal relevances compute relevance of other paths
        start_layer_ = "y" if not start_layer else start_layer

        [condition.update(dict(
            zip(
                [k for k in condition.keys() if (k != start_layer_)],
                [[] for k, v in condition.items() if (k != start_layer_)]
            ))) for condition in multiple_conditions]

        model = self.model
        self.model = self.model_copy
        attr_result_ = CondAttribution.__call__(self, copy.copy(data), multiple_conditions, composite, record_layer,
                                                mask_map, start_layer, init_rel, on_device)
        self.model = model
        # compute relevance differences to only get conditional relevance
        mult_conditions = [(len(condition) > 1) for condition in conditions]
        attr_result.heatmap[mult_conditions] -= attr_result_.heatmap
        for k in attr_result.relevances.keys():
            attr_result.relevances[k][mult_conditions] -= attr_result_.relevances[k]
        return attr_result


class EpsilonGammaFlat(SpecialFirstLayerMapComposite):
    '''An explicit composite using the flat rule for any linear first layer, the zplus rule for all other convolutional
    layers and the epsilon rule for all other fully connected layers.
    '''
    def __init__(self, canonizers=None):
        layer_map = LAYER_MAP_BASE + [
            (Convolution, Gamma(0.1)),
            (torch.nn.Linear, Epsilon()),
        ]
        first_map = [
            (Linear, Flat())
        ]
        super().__init__(layer_map, first_map, canonizers=canonizers)