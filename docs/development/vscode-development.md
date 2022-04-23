# VScode containerised remote ssh development
source `https://stackoverflow.com/a/63617958/1312802`
1. Install the Remote-Containers extension

![Remote-Containers](../images/remote-containers-extensions.png)

2. Open the docker.host setting in the your `setting.json`

![Docker Host Setting](../images/docker-host-setting.png)

2. Change the server here to the user and ssh remote machine address
   `ssh://your-remote-user@your-remote-machine-fqdn-or-ip-here`

![Docker Host ssh](../images/docker-host-ssh.png)
   
3. ssh into your remote server and set up a container based on the image you have built before, also don't forget to bind your project volume to the container instance.
   ```
   docker run -d -it --name devcontainer --mount type=bind,source="$(pwd)"/target,target=/app devimage
   ```

4. Come back to the VScode and open command palette and select following option:

![Attach to the container](../images/attach-to-remote-container.png)

5. Select your remote container on the server

![Select the container](../images/select-remote-container.png)


You are good to go!
   