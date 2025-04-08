import axios from 'axios';
import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import { isKaggleEnvironmentsEpisode, parseKaggleEnvironmentsEpisode } from './episode/kaggle-environments';
import { isLuxAISEpisode, parseLuxAISEpisode } from './episode/luxai';
import { Episode, Tile } from './episode/model';
import { DisplayConfig, FogOfWar } from './pages/visualizer/Board';

const PRODUCTION_BASE_URL = 'https://s3vis.lux-ai.org';

export interface State {
  episode: Episode | null;
  rawEpisode: any | null;

  turn: number;
  speed: number;

  selectedTile: Tile | null;
  scrollSelectedTileToTop: boolean;

  loading: boolean;
  progress: number;

  minimalTheme: boolean;

  displayConfig: DisplayConfig;

  setTurn: (turn: number) => void;
  increaseTurn: () => boolean;
  setSpeed: (speed: number) => void;
  setSelectedTile: (selectedTile: Tile | null, scrollSelectedTileToTop: boolean) => void;

  load: (data: any) => void;
  loadFromFile: (file: File) => Promise<void>;
  loadFromInput: (input: string, proxy: string) => Promise<void>;

  openInNewTab: () => void;

  setTheme: (minimal: boolean) => void;
  setDisplayConfig: (displayConfig: DisplayConfig) => void;
}

export const useStore = create(
  persist<State>(
    (set, get) => ({
      episode: null,

      rawEpisode: null,

      turn: 1,
      speed: 1,

      selectedTile: null,
      scrollSelectedTileToTop: false,

      loading: false,
      progress: 0,
      displayConfig: {
        sensorMask: true,
        energyField: false,
        relicConfigs: true,
        fogOfWar: FogOfWar.None,
      },
      minimalTheme: true,

      setTurn: turn => {
        if (get().turn !== turn) {
          set({ turn });
        }
      },

      increaseTurn: () => {
        if (get().turn === get().episode!.steps.length - 1) {
          return false;
        }

        set(state => ({ turn: state.turn + 1 }));
        return true;
      },

      setSpeed: speed => {
        if (get().speed !== speed) {
          set({ speed });
        }
      },

      setSelectedTile: (selectedTile, scrollSelectedTileToTop) => {
        const { selectedTile: currentSelectedTile, scrollSelectedTileToTop: currentScrollSelectedTileToTop } = get();
        if (
          (selectedTile === null && currentSelectedTile !== null) ||
          (selectedTile !== null &&
            (selectedTile.x !== currentSelectedTile?.x || selectedTile.y !== currentSelectedTile?.y)) ||
          scrollSelectedTileToTop !== currentScrollSelectedTileToTop
        ) {
          set({ selectedTile, scrollSelectedTileToTop });
        }
      },

      load: data => {
        const formatError =
          'Episode data has unsupported format, only HTML and JSON replays generated using the luxai-s3 CLI and JSON replays generated using the kaggle-environments CLI are supported';

        if (typeof data !== 'object') {
          if (!data.startsWith('{')) {
            const matches = /window\.episode = (.*);/.exec(data);
            if (matches === null) {
              throw new Error(formatError);
            }

            data = matches[1];
          }

          try {
            data = JSON.parse(data);
          } catch (err) {
            throw new Error(formatError);
          }
        }

        let episode: Episode | null = null;
        if (isLuxAISEpisode(data)) {
          episode = parseLuxAISEpisode(data);
        } else if (isKaggleEnvironmentsEpisode(data)) {
          episode = parseKaggleEnvironmentsEpisode(data);
        } else {
          throw new Error(formatError);
        }

        set({
          episode,
          rawEpisode: data,
          turn: 0,
          speed: 1,
        });
      },

      loadFromFile: file => {
        return new Promise((resolve, reject) => {
          set({ loading: true, progress: 0 });

          const reader = new FileReader();

          reader.addEventListener('load', () => {
            set({ loading: false });

            try {
              get().load(reader.result as string);
              resolve();
            } catch (err: any) {
              reject(err);
            }
          });

          reader.addEventListener('error', () => {
            reject(new Error('FileReader emitted an error event'));
          });

          reader.readAsText(file);
        });
      },

      loadFromInput: async (input, proxy) => {
        set({ loading: true, progress: 0 });

        const interestingPrefixes = [
          'https://www.kaggle.com/competitions/lux-ai-season-3/leaderboard?dialog=episodes-episode-',
          'https://www.kaggle.com/competitions/lux-ai-season-3/submissions?dialog=episodes-episode-',
        ];

        let url: string;
        if (/^\d+$/.test(input)) {
          url = `https://www.kaggleusercontent.com/episodes/${input}.json`;
        } else if (interestingPrefixes.some(prefix => input.startsWith(prefix))) {
          const id = input.split('-').pop();
          url = `https://www.kaggleusercontent.com/episodes/${id}.json`;
        } else {
          url = input;
        }

        let parsedURL: URL;
        try {
          parsedURL = new URL(url);
        } catch (err: any) {
          set({ loading: false });
          throw new Error('Invalid input');
        }

        if (
          parsedURL.hostname !== 'localhost' &&
          parsedURL.origin !== window.location.origin &&
          proxy.trim().length > 0
        ) {
          url = proxy + url;
        }

        try {
          const response = await axios.get(url, {
            onDownloadProgress: event => {
              if (event.loaded && event.total) {
                set({ progress: event.loaded / event.total });
              }
            },
          });

          set({ loading: false, progress: 0 });
          get().load(response.data);
        } catch (err: any) {
          set({ loading: false, progress: 0 });

          console.error(err);

          if (
            err.response &&
            typeof err.response.data === 'string' &&
            err.response.data.endsWith('was not whitelisted by the operator of this proxy.')
          ) {
            throw new Error(
              'The current origin is not whitelisted by the operator of the specified CORS Anywhere proxy',
            );
          }

          throw new Error(`${err.message}, see the browser console for more information`);
        }
      },

      openInNewTab: () => {
        const tab = window.open(`${PRODUCTION_BASE_URL}/#/open`, '_blank')!;
        for (const ms of [100, 250, 500, 750, 1000, 1500, 2000, 3000, 4000, 5000, 10000]) {
          setTimeout(() => tab.postMessage(get().rawEpisode, PRODUCTION_BASE_URL), ms);
        }
      },

      setTheme: minimal => {
        if (get().minimalTheme !== minimal) {
          set({ minimalTheme: minimal });
        }
      },
      setDisplayConfig: (displayConfig: DisplayConfig) => {
        set({ displayConfig });
      },
      setFogOfWar: (fogOfWar: FogOfWar) => {
        set(state => ({ displayConfig: { ...state.displayConfig, fogOfWar } }));
      },
    }),
    {
      name: 'lux-eye-s3',
      partialize: state =>
        ({
          minimalTheme: state.minimalTheme,
        } as any),
    },
  ),
);
