import logging
from gymnasium.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='SimplePacman-v0',
    entry_point='gym_envs.envs.gym_pacman.custom_pacman_env:PacmanEnv',
    kwargs={
        'config_file': ""
    }
)

register(
    id='DungeonCrawler-v0',
    entry_point='gym_envs.envs.gym_dungeon.custom_dungeon_env:DungeonCrawlerEnv',
    kwargs={
        'config_file': ""
    }
)

register(
    id='SimpleMinecraft-v0',
    entry_point='gym_envs.envs.gym_minecraft.custom_minecraft_env:MinecraftEnv',
)

register(
    id='SimpleSaladChef-v0',
    entry_point='gym_envs.envs.gym_saladchef.custom_saladchef_env:SaladChefEnv',
)
