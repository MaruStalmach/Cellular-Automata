from .. import Rule, State, Geometry
import numpy as np

class Diffusion(Rule):
    """Random walk diffusion based on "Quantitative Cellular Automaton Model For Biofilms" by Pizarro G.

    arguments:
    a - defined as p0/p3, where p0 is probability that a particle wont change direction and p3 is probability that a particle will make a 180 deg turn
    p - probability for a particle to do a 90 deg turn. p=p1=p2=p4=p5 because of symmetry.

    """

    def __init__(
        self, geometry: Geometry, a: float, p: float, target_key: str = "substrate"
    ):
        super().__init__(geometry)

        self.target_key = target_key
        p3 = (1 - 4 * p) / (a + 1)
        self.p = [a * p3, p, p, p3, p, p]

    def apply_state(self, state) -> State:
        ###TODO: force dtype at layers initialization
        layers = np.empty(shape=state.shape + (6,))
        layers[...] = state[self.target_key]
        # layers = np.zeros_like(layers)

        # each layer corresponds to the dircetion a particle is moving, list p contains probablities of the next direction
        # p0 - no change, p3 - 180 deg turn, p1,p2,p4,p5 - 90 degree turns

        # layer 0 - +z
        # l1      - +y
        # l2      - +x
        # l3      - -z
        # l4      - -y
        # l5      - -x

        layers_shape = layers.shape
        num_particles = layers.sum()
        # breakpoint()

        rands = np.random.choice([0, 1, 2, 3, 4, 5], size=int(num_particles), p=self.p)
        rolls = np.zeros(shape=layers_shape)

        rolls = rolls.flatten()
        layers = layers.flatten()

        rolls[layers == 1] = rands

        layers = layers.reshape(layers_shape)
        rolls = rolls.reshape(layers_shape)

        h = np.stack(
            [
                np.full(shape=state.shape, fill_value=x, dtype=np.uint8)
                for x in range(6)
            ],
            axis=-1,
        )

        rolls = (rolls + h) % 6
        # set empty cells as invalid value
        rolls[layers == 0] = 6

        # reset layers to zeros
        layers[...] = 0

        # does check for collisions
        for i in range(6):
            r2 = np.zeros_like(rolls)
            r2[rolls == i] = 1
            layers[..., i] = np.add.reduce(r2, axis=-1)

        # #doesnt check for collisions, some particles disappear
        # for i in range(6):
        #     layers[np.any(rolls==i,axis=-1),i]=1

        # move particles according to their movement direction (corresponding layer)
        # layer 0 - +z
        # l1      - +y
        # l2      - +x
        # l3      - -z
        # l4      - -y
        # l5      - -x

        ## TODO: maybe instead of this reroll edge particles with p_0=0
        for i, ax_dir in enumerate([(2, 1), (1, 1), (0, 1)]):
            ax, dir = ax_dir
            if ax not in state.geometry.periodic_dims:
                sel = [slice(None)] * 3
                # save the edge next to wall and zero it in the array so that when it rolls later zeros come out on the other side
                sel[ax] = -1
                edge = layers[tuple(sel) + (i,)].copy()
                layers[tuple(sel) + (i,)] = 0
                # put the edge values on -2 as if they bounced
                sel[ax] = -2
                layers[tuple(sel) + (i,)] += edge
                # similarly for the opposite direction
                sel[ax] = 0
                edge = layers[tuple(sel) + (i + 3,)].copy()
                layers[tuple(sel) + (i + 3,)] = 0
                # put the edge values on 1 as if they bounced
                sel[ax] = 1
                layers[tuple(sel) + (i + 3,)] += edge

            layers[..., i] = np.roll(layers[..., i], dir, ax)
            layers[..., i + 3] = np.roll(layers[..., i + 3], dir * -1, ax)
            # TODO: deal with collsions >1 values in arrays need to be spread out or some shit idk

        # naive collision resolution -> delete colliding particles
        layers[layers > 1] = 1

        state[self.target_key] = layers[...]
        return state