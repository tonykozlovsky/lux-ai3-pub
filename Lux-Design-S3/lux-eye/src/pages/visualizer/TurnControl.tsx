import { ActionIcon, Button, Group, Kbd, Menu, Slider, Stack, Table, Text } from '@mantine/core';
import { HotkeyItem, useElementSize, useHotkeys } from '@mantine/hooks';
import { useModals } from '@mantine/modals';
import {
  IconArrowUpRight,
  IconChevronsLeft,
  IconChevronsRight,
  IconInfoCircle,
  IconKeyboard,
  IconPlayerPause,
  IconPlayerPlay,
  IconPlayerTrackNext,
  IconPlayerTrackPrev,
} from '@tabler/icons';
import { useCallback, useEffect, useMemo, useState } from 'react';
import { getMatchIdx, parseTileType } from '../../episode/luxai';
import { useStore } from '../../store';
import { FogOfWar } from './Board';

const SPEEDS = [0.5, 1, 2, 4, 8, 16, 32];

interface TurnControlProps {
  showHotkeysButton: boolean;
  showOpenButton: boolean;
}

export function TurnControl({ showHotkeysButton, showOpenButton }: TurnControlProps): JSX.Element {
  const episode = useStore(state => state.episode)!;
  const turn = useStore(state => state.turn);

  const setTurn = useStore(state => state.setTurn);
  const increaseTurn = useStore(state => state.increaseTurn);

  const speed = useStore(state => state.speed);
  const setSpeed = useStore(state => state.setSpeed);

  const selectedTile = useStore(state => state.selectedTile);

  const openInNewTab = useStore(state => state.openInNewTab);

  const displayConfig = useStore(state => state.displayConfig);
  const setDisplayConfig = useStore(state => state.setDisplayConfig);

  /* const minimalTheme = useStore(state => state.minimalTheme);
  const setTheme = useStore(state => state.setTheme); */

  const [playing, setPlaying] = useState(false);

  const modals = useModals();

  const { ref: sliderRef, width: sliderWidth } = useElementSize();

  const sliderStyles = useMemo(() => {
    const padding = 10;
    const pxPerStep = (sliderWidth - padding * 2) / episode.steps.length;

    const gradientParts: [string, number, number][] = [];

    for (const step of episode.steps) {
      const isDay = getMatchIdx(step.step, episode.params) % 2 === 0;
      const color = isDay ? '#e9ecef' : '#bdc3c7';

      if (gradientParts.length === 0) {
        gradientParts.push([color, padding, padding + pxPerStep]);
      } else {
        const lastPart = gradientParts[gradientParts.length - 1];
        if (lastPart[0] === color) {
          lastPart[2] += pxPerStep;
        } else {
          gradientParts.push([color, lastPart[2], lastPart[2] + pxPerStep]);
        }
      }
    }

    let sliderEnd = pxPerStep * (episode.steps.length + 1);
    if (gradientParts.length > 0) {
      const lastGradientPart = gradientParts[gradientParts.length - 1];
      lastGradientPart[2] = Math.floor(lastGradientPart[2]);
      sliderEnd = lastGradientPart[2];
    }

    const gradientArgs = gradientParts.map(parts => `${parts[0]} ${parts[1]}px ${parts[2]}px`);
    gradientArgs.push(`#228be6 ${sliderEnd}px ${sliderWidth + padding * 2}px`);

    return {
      track: {
        // eslint-disable-next-line @typescript-eslint/naming-convention
        '&:before': {
          background: `linear-gradient(to right, ${gradientArgs.join(', ')})`,
        },
      },
      bar: {
        borderTopRightRadius: '0px',
        borderBottomRightRadius: '0px',
      },
      thumb: {
        width: '1px',
        border: '1px solid black',
        borderRadius: '0px',
        marginLeft: '-1px',
      },
    };
  }, [episode, sliderWidth]);

  const onSliderChange = useCallback((value: number) => {
    setTurn(value - 1);
    setPlaying(false);
  }, []);

  const togglePlaying = useCallback(() => {
    if (!playing && turn === episode.steps.length - 1) {
      return;
    }

    setPlaying(!playing);
  }, [episode, turn, playing]);

  const toggleEnergyFieldDisplay = useCallback(() => {
    setDisplayConfig({
      ...displayConfig,
      energyField: !displayConfig.energyField,
    });
  }, [displayConfig]);

  const toggleRelicConfigDisplay = useCallback(() => {
    setDisplayConfig({
      ...displayConfig,
      relicConfigs: !displayConfig.relicConfigs,
    });
  }, [displayConfig]);

  const toggleSensorMaskDisplay = useCallback(() => {
    setDisplayConfig({
      ...displayConfig,
      sensorMask: !displayConfig.sensorMask,
    });
  }, [displayConfig]);

  const setFogOfWar = useCallback(
    (fogOfWar: FogOfWar) => {
      setDisplayConfig({
        ...displayConfig,
        fogOfWar,
      });
    },
    [displayConfig],
  );

  const previousTurn = useCallback(() => {
    if (turn > 0 && !sliderRef.current?.contains(document.activeElement)) {
      setTurn(turn - 1);
    }

    setPlaying(false);
  }, [episode, turn, sliderRef]);

  const nextTurn = useCallback(() => {
    if (turn < episode.steps.length - 1 && !sliderRef.current?.contains(document.activeElement)) {
      setTurn(turn + 1);
    }

    setPlaying(false);
  }, [episode, turn, sliderRef]);

  const increaseSpeed = useCallback(() => {
    setSpeed(SPEEDS[Math.min(SPEEDS.indexOf(speed) + 1, SPEEDS.length - 1)]);
  }, [speed]);

  const decreaseSpeed = useCallback(() => {
    setSpeed(SPEEDS[Math.max(SPEEDS.indexOf(speed) - 1, 0)]);
  }, [speed]);

  const goToStart = useCallback(() => {
    setTurn(0);
  }, [episode]);

  const goToEnd = useCallback(() => {
    setTurn(episode.steps.length - 1);
  }, [episode]);

  /* const toggleTheme = useCallback(() => {
    setTheme(!minimalTheme);
  }, [minimalTheme]); */

  const openHotkeysModal = useCallback(() => {
    modals.openModal({
      title: 'Hotkeys',
      children: (
        <Table>
          <thead>
            <tr>
              <th>Hotkey</th>
              <th>Description</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>
                <Kbd>Space</Kbd>
              </td>
              <td>Play/pause</td>
            </tr>
            <tr>
              <td>
                <Kbd>◄</Kbd>
              </td>
              <td>Previous turn</td>
            </tr>
            <tr>
              <td>
                <Kbd>►</Kbd>
              </td>
              <td>Next turn</td>
            </tr>
            <tr>
              <td>
                <Kbd>▲</Kbd>
              </td>
              <td>Increase speed</td>
            </tr>
            <tr>
              <td>
                <Kbd>▼</Kbd>
              </td>
              <td>Decrease speed</td>
            </tr>
            <tr>
              <td>
                <Kbd>e</Kbd>
              </td>
              <td>Toggle Energy Field Display</td>
            </tr>
            <tr>
              <td>
                <Kbd>r</Kbd>
              </td>
              <td>Toggle Relic Config Display</td>
            </tr>
            <tr>
              <td>
                <Kbd>s</Kbd>
              </td>
              <td>Toggle Sensor Mask Display</td>
            </tr>
            {SPEEDS.map((speed, i) => (
              <tr key={i}>
                <td>
                  <Kbd>{i + 1}</Kbd>
                </td>
                <td>Set speed to {speed}x</td>
              </tr>
            ))}
          </tbody>
        </Table>
      ),
    });
  }, []);

  const openJsonDataModal = useCallback(() => {
    modals.openModal({
      title: "This Game's Parameters",
      size: 'auto',
      children: <pre>{JSON.stringify(episode.params, null, 2)}</pre>,
    });
  }, [episode]);

  useEffect(() => {
    if (!playing) {
      return;
    }

    const interval = setInterval(() => {
      if (!increaseTurn()) {
        setPlaying(false);
      }
    }, 1000 / speed);

    return () => clearInterval(interval);
  }, [playing, speed]);

  const hotkeys: HotkeyItem[] = [
    ['space', togglePlaying],
    ['ArrowLeft', previousTurn],
    ['ArrowRight', nextTurn],
    ['ArrowUp', increaseSpeed],
    ['ArrowDown', decreaseSpeed],
    ['e', toggleEnergyFieldDisplay],
    ['r', toggleRelicConfigDisplay],
    ['s', toggleSensorMaskDisplay],
  ];

  for (let i = 1; i <= SPEEDS.length; i++) {
    hotkeys.push([i.toString(), () => setSpeed(SPEEDS[i - 1])]);
  }

  useHotkeys(hotkeys);

  const step = episode.steps[turn];
  const matchIdx = getMatchIdx(step.step, episode.params);
  return (
    <Stack spacing="xs">
      <Slider
        ref={sliderRef}
        min={1}
        max={episode.steps.length}
        onChange={onSliderChange}
        value={turn + 1}
        label={null}
        size="lg"
        // radius={0}
        styles={sliderStyles}
      />

      <Group>
        <ActionIcon color="blue" variant="transparent" title="Go to start" onClick={goToStart}>
          <IconPlayerTrackPrev />
        </ActionIcon>
        <ActionIcon color="blue" variant="transparent" title="Slower" onClick={decreaseSpeed}>
          <IconChevronsLeft />
        </ActionIcon>
        <ActionIcon color="blue" variant="transparent" title={playing ? 'Pause' : 'Play'} onClick={togglePlaying}>
          {playing ? <IconPlayerPause /> : <IconPlayerPlay />}
        </ActionIcon>
        <ActionIcon color="blue" variant="transparent" title="Faster" onClick={increaseSpeed}>
          <IconChevronsRight />
        </ActionIcon>
        <ActionIcon color="blue" variant="transparent" title="Go to end" onClick={goToEnd}>
          <IconPlayerTrackNext />
        </ActionIcon>
        {/* <ActionIcon color="blue" variant="transparent" title="Toggle theme (minimal/aesthetic)" onClick={toggleTheme}>
          {minimalTheme ? <IconBrushOff /> : <IconBrush />}
        </ActionIcon> */}
        {showHotkeysButton && (
          <ActionIcon color="blue" variant="transparent" title="Show hotkeys" onClick={openHotkeysModal}>
            <IconKeyboard />
          </ActionIcon>
        )}
        {showOpenButton && (
          <ActionIcon color="blue" variant="transparent" title="Open in new tab" onClick={openInNewTab}>
            <IconArrowUpRight />
          </ActionIcon>
        )}
        <ActionIcon color="blue" variant="transparent" title="Show JSON data" onClick={openJsonDataModal}>
          <IconInfoCircle />
        </ActionIcon>
        <Menu shadow="md" width={200}>
          <Menu.Target>
            <Button color="blue" variant={displayConfig.fogOfWar == FogOfWar.None ? 'outline' : 'filled'} size="xs">
              FOW
            </Button>
          </Menu.Target>
          <Menu.Dropdown>
            <Menu.Label>Set Fog of War</Menu.Label>
            <Menu.Item onClick={() => setFogOfWar(FogOfWar.None)}>None</Menu.Item>
            <Menu.Item onClick={() => setFogOfWar(FogOfWar.Team1)}>Team 1</Menu.Item>
            <Menu.Item onClick={() => setFogOfWar(FogOfWar.Team2)}>Team 2</Menu.Item>
            <Menu.Item onClick={() => setFogOfWar(FogOfWar.Both)}>Both</Menu.Item>
          </Menu.Dropdown>
        </Menu>

        <div style={{ marginRight: 'auto' }} />

        <Text>{speed}x</Text>
        <Text>
          {episode.steps[turn].step} / {episode.steps[episode.steps.length - 1].step}
        </Text>
      </Group>

      <Group position="apart">
        <Text>{`Match ${matchIdx + 1} / ${episode.params.match_count_per_episode}`}</Text>
        {episode.metadata.seed && <Text>Seed: {episode.metadata.seed}</Text>}
        {selectedTile !== null && (
          <Text>
            Tile: ({selectedTile.x}, {selectedTile.y})
          </Text>
        )}
      </Group>

      {selectedTile !== null && (
        <Group position="apart">
          <Text>
            Tile Type: {parseTileType(step.board.tileType[selectedTile.x][selectedTile.y])}{' '}
            {step.board.energyNodes.find(val => {
              return val[0] == selectedTile.x && val[1] == selectedTile.y;
            }) && '(Energy Node)'}{' '}
            {step.board.relicNodes.find(val => {
              return val[0] == selectedTile.x && val[1] == selectedTile.y;
            }) && '(Relic Node)'}
          </Text>

          <Text>Energy: {step.board.energy[selectedTile.x][selectedTile.y]}</Text>
          <Text>
            Vision Power: {step.board.visionPowerMap[0][selectedTile.x][selectedTile.y]},{' '}
            {step.board.visionPowerMap[1][selectedTile.x][selectedTile.y]}
          </Text>
        </Group>
      )}

      {selectedTile === null && <Text align="right">No tile selected</Text>}
    </Stack>
  );
}
